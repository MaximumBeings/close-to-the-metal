# Appendix F: Production Configuration Templates

> *"Copy-paste infrastructure is dangerous. Copy-paste-understand-then-modify infrastructure is how real systems get built."*

---

This appendix provides complete, production-tested configuration templates for common deployment scenarios. Each template includes inline comments explaining why each value was chosen.

---

## F.1 Single-GPU vLLM Deployment

### F.1.1 Single A100 80GB — Llama-3.1-8B-Instruct

```bash
#!/bin/bash
# deploy_8b_a100.sh
#
# Target: Interactive chat API, <500ms TTFT, ~50 concurrent users
# Hardware: 1× A100 80GB SXM
#
# Key flag choices:
#   --dtype bfloat16               A100 supports BF16 natively
#   --max-model-len 8192           8K context (saves 4× KV vs 32K)
#   --gpu-memory-utilization 0.88  leave 12% headroom for activations
#   --kv-cache-dtype fp8           2× KV capacity vs BF16
#   --swap-space 8                 8GB CPU swap for preemption
#   --max-num-seqs 256             up to 256 concurrent sequences
#   --enable-chunked-prefill       prevent long prefills from blocking decode
#   --max-num-batched-tokens 2048  prefill chunk size (2K tokens/step)
#   --scheduler-delay-factor 0.0   prioritize TTFT (interactive)
#   --enable-prefix-caching        cache shared system prompt prefix
#   --disable-log-requests         reduce log volume in production
#   --max-log-len 100              truncate logged prompts for privacy

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-dtype fp8 \
    --swap-space 8 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --scheduler-delay-factor 0.0 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-requests \
    --max-log-len 100
```

### F.1.2 Single RTX 4090 24GB — Qwen2.5-7B-Instruct

```bash
#!/bin/bash
# deploy_7b_4090.sh
#
# Target: Development/staging environment, moderate load
# Hardware: 1× RTX 4090 24GB (Ada Lovelace, SM89)
#
# Key flag choices:
#   --max-model-len 16384           16K context (model default is 128K but costs KV)
#   --gpu-memory-utilization 0.85   more conservative on consumer GPU

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --swap-space 4 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000
```

---

## F.2 Multi-GPU vLLM Deployment

### F.2.1 4× H100 80GB — Llama-3.1-70B-Instruct

```bash
#!/bin/bash
# deploy_70b_4xh100.sh
#
# Target: Production API, high concurrency, <800ms TTFT at p95
# Hardware: 4× H100 80GB NVLink
#
# Key flag choices:
#   --tensor-parallel-size 4        split across 4 H100s
#   --max-model-len 32768           32K context
#   --gpu-memory-utilization 0.90   4× 80GB = 320GB; ~140GB weights, ~180GB KV
#   --kv-cache-dtype fp8            2× KV capacity: ~360GB equivalent
#   --swap-space 16                 per-node CPU swap
#   --scheduler-delay-factor 0.1    small delay to build larger decode batches

vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8 \
    --swap-space 16 \
    --max-num-seqs 512 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --scheduler-delay-factor 0.1 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-requests \
    --uvicorn-log-level warning
```

### F.2.2 8× H200 141GB — DeepSeek-V3

```bash
#!/bin/bash
# deploy_deepseek_v3_8xh200.sh
#
# Target: Frontier-model quality production API
# Hardware: 8× H200 141GB NVLink (1,128GB total)
#
# Key flag choices:
#   --gpu-memory-utilization 0.88   conservative with 671B model
#   --swap-space 32                 32GB swap per node
#   --trust-remote-code             DeepSeek uses custom MoE code

vllm serve deepseek-ai/DeepSeek-V3 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-dtype fp8 \
    --swap-space 32 \
    --max-num-seqs 128 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-requests
```

---

## F.3 llama.cpp Production Templates

### F.3.1 CPU-Only Server (Edge Deployment)

```bash
#!/bin/bash
# deploy_cpu_server.sh
#
# Target: Edge deployment, no GPU, low-latency single-user
# Hardware: 32-core server, 128GB RAM
#
# Key flag choices:
#   -t 16       16 threads for generation
#   -tb 32      32 threads for prompt processing (batch/prefill phase)
#   -c 8192     8K context window
#   -np 4       4 parallel slots (concurrent users)
#   -b 512      conservative batch size for CPU
#   --no-mmap   pre-load entire model into RAM (consistent latency)
#   --mlock     lock model in RAM (prevent swapping under load)

llama-server \
    -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -t 16 \
    -tb 32 \
    -c 8192 \
    -np 4 \
    -b 512 \
    --no-mmap \
    --mlock \
    --temp 0.7 \
    --top-k 40 \
    --top-p 0.95 \
    --chat-template qwen \
    -cb \
    --host 0.0.0.0 \
    --port 8080 \
    --timeout 300
```

### F.3.2 GPU Server (Single Consumer GPU)

```bash
#!/bin/bash
# deploy_gpu_server.sh
#
# Target: Home lab or small team, RTX 3090/4090
# Hardware: RTX 4090 24GB
#
# Key flag choices:
#   -ngl 99     offload all layers to GPU (99 = "all")
#   -c 16384    16K context window
#   -np 8       8 parallel slots (concurrent users)
#   -b 2048     batch size for prefill phase
#   -ub 512     micro-batch size for decode phase

llama-server \
    -m ./models/llama-3.1-8b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -c 16384 \
    -np 8 \
    -b 2048 \
    -ub 512 \
    --chat-template llama3 \
    -cb \
    --host 0.0.0.0 \
    --port 8080
```

### F.3.3 High-Memory Server (Multi-GPU llama.cpp)

```bash
#!/bin/bash
# deploy_multigpu_llamacpp.sh
#
# Target: 70B model split across 2× GPUs
# Hardware: 2× RTX 3090 24GB = 48GB total
#
# Key flag choices:
#   -sm layer   split by layer across GPUs (row/none are alternatives)
#   -ts 1,1     equal tensor split across GPUs
#   -mg 0       main GPU is device 0 (receives final logits)
#   -c 4096     conservative context — 70B is memory intensive

llama-server \
    -m ./models/llama-3.1-70b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -sm layer \
    -ts 1,1 \
    -mg 0 \
    -c 4096 \
    -np 4 \
    -b 2048 \
    --chat-template llama3 \
    -cb \
    --host 0.0.0.0 \
    --port 8080
```

---

## F.4 Kubernetes / KubeRay Deployment

### F.4.1 KubeRay Cluster Template

```yaml
# ray-cluster.yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: vllm-cluster
spec:
  rayVersion: '2.32.0'
  
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray-ml:2.32.0-gpu
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "32Gi"
            requests:
              nvidia.com/gpu: "1"
              memory: "32Gi"
          env:
          - name: NCCL_IB_DISABLE
            value: "1"
  
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 3         # 3 worker nodes
    minReplicas: 1
    maxReplicas: 8      # autoscale up to 8
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray-ml:2.32.0-gpu
          resources:
            limits:
              nvidia.com/gpu: "4"   # 4 GPUs per worker
              memory: "256Gi"
            requests:
              nvidia.com/gpu: "4"
              memory: "256Gi"
          env:
          - name: NCCL_IB_DISABLE
            value: "1"
          volumeMounts:
          - name: model-cache
            mountPath: /root/.cache/huggingface
        volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
```

### F.4.2 vLLM Deployment on KubeRay

```python
# vllm_serve_ray.py
from ray import serve
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import FastAPI
import ray

app = FastAPI()

@serve.deployment(
    name="vllm-llama-70b",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 4,                  # 4 GPUs per replica
        "num_cpus": 8,
        "memory": 64 * 1024 ** 3,      # 64GB RAM
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 32,  # scale up when > 32 requests in flight
    },
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self):
        engine_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.1-70B-Instruct",
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=32768,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=True,
            kv_cache_dtype="fp8",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/completions")
    async def completions(self, request: dict):
        from vllm import SamplingParams
        sampling = SamplingParams(
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 512),
        )
        async for output in self.engine.generate(
            request["prompt"], sampling, request_id="req"
        ):
            if output.finished:
                return {"text": output.outputs[0].text}

deployment = VLLMDeployment.bind()
```

---

## F.5 Load Balancer Configuration

### F.5.1 nginx Configuration

```nginx
# /etc/nginx/conf.d/vllm.conf
#
# Load balance across multiple vLLM instances

upstream vllm_backends {
    least_conn;           # route to backend with fewest connections
    
    server 10.0.0.1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.0.2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.0.3:8000 weight=1 max_fails=3 fail_timeout=30s;
    
    keepalive 64;         # maintain connection pool
}

server {
    listen 80;
    server_name llm-api.example.com;
    
    location /v1/ {
        proxy_pass http://vllm_backends;
        
        # Streaming support (for SSE)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;  # long timeout for streaming responses
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Connection settings
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    location /health {
        proxy_pass http://vllm_backends/health;
        proxy_read_timeout 5s;
    }
}
```

### F.5.2 Envoy Proxy Configuration (Advanced)

```yaml
# envoy.yaml
# Envoy proxy with health checking, circuit breaking, and retries

static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 8080
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          route_config:
            virtual_hosts:
            - name: vllm
              domains: ["*"]
              routes:
              - match:
                  prefix: "/v1/"
                route:
                  cluster: vllm_cluster
                  timeout: 300s
                  retry_policy:
                    retry_on: "5xx,reset,connect-failure"
                    num_retries: 2
  
  clusters:
  - name: vllm_cluster
    connect_timeout: 5s
    type: ROUND_ROBIN
    circuit_breakers:
      thresholds:
      - max_connections: 1000
        max_pending_requests: 500
        max_requests: 2000
    health_checks:
    - timeout: 5s
      interval: 10s
      unhealthy_threshold: 3
      healthy_threshold: 2
      http_health_check:
        path: "/health"
    load_assignment:
      cluster_name: vllm_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.0.0.1
                port_value: 8000
        - endpoint:
            address:
              socket_address:
                address: 10.0.0.2
                port_value: 8000
```

---

## F.6 Monitoring Stack

### F.6.1 Prometheus Scrape Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets:
        - '10.0.0.1:8000'
        - '10.0.0.2:8000'
    metrics_path: '/metrics'
    
  - job_name: 'node_exporter'
    static_configs:
      - targets:
        - '10.0.0.1:9100'
        - '10.0.0.2:9100'
        
  - job_name: 'dcgm_exporter'   # NVIDIA GPU metrics
    static_configs:
      - targets:
        - '10.0.0.1:9400'
        - '10.0.0.2:9400'
```

### F.6.2 Key Grafana Dashboard Queries

```promql
# Time to First Token (P95)
histogram_quantile(0.95,
  rate(vllm:time_to_first_token_seconds_bucket[5m])
)

# Inter-Token Latency (P95)
histogram_quantile(0.95,
  rate(vllm:time_per_output_token_seconds_bucket[5m])
)

# GPU Utilization
avg(DCGM_FI_DEV_GPU_UTIL) by (instance)

# KV Cache Hit Rate (prefix caching)
rate(vllm:gpu_prefix_cache_hits_total[5m]) /
(rate(vllm:gpu_prefix_cache_hits_total[5m]) + rate(vllm:gpu_prefix_cache_misses_total[5m]))

# Request Queue Length
vllm:num_requests_waiting

# Tokens per Second
rate(vllm:generation_tokens_total[1m])

# GPU Memory Usage (%)
vllm:gpu_cache_usage_perc
```

---

## F.7 Systemd Service Files

### F.7.1 vLLM systemd Service

```ini
# /etc/systemd/system/vllm.service

[Unit]
Description=vLLM Inference Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=inference
Group=inference
WorkingDirectory=/opt/vllm

# Environment
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3"
Environment="NCCL_IB_DISABLE=1"
EnvironmentFile=/etc/vllm/env

# Command
ExecStart=/opt/vllm/venv/bin/vllm serve \
    meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --port 8000

# Restart policy
Restart=on-failure
RestartSec=30s
StartLimitInterval=5min
StartLimitBurst=3

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vllm

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm

# Monitor
sudo journalctl -u vllm -f
sudo systemctl status vllm
```

### F.7.2 llama-server systemd Service

```ini
# /etc/systemd/system/llamacpp.service

[Unit]
Description=llama.cpp Inference Server
After=network.target

[Service]
Type=simple
User=inference
WorkingDirectory=/opt/llamacpp

ExecStart=/opt/llamacpp/llama-server \
    -m /opt/models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -c 16384 \
    -np 8 \
    --chat-template qwen \
    -cb \
    --host 0.0.0.0 \
    --port 8080

Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

---

## F.8 Environment Variables Reference

```bash
# CUDA and GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3    # which GPUs to use
export CUDA_LAUNCH_BLOCKING=1          # debug: synchronous CUDA (slow)

# NCCL (multi-GPU communication)
export NCCL_IB_DISABLE=1              # disable InfiniBand (if unavailable)
export NCCL_P2P_LEVEL=NVL             # NVLink P2P
export NCCL_DEBUG=INFO                # verbose NCCL logging (debug)
export NCCL_SOCKET_IFNAME=eth0        # network interface for NCCL

# vLLM-specific
export VLLM_ATTENTION_BACKEND=FLASHINFER    # use FlashInfer kernels
export VLLM_USE_TRITON_FLASH_ATTN=1         # use Triton Flash Attention
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # process spawning method
export VLLM_TRACE_FUNCTION=1                # trace function calls (debug)

# Hugging Face
export HF_HOME=/data/huggingface          # model cache directory
export HUGGING_FACE_HUB_TOKEN=hf_xxx     # auth token for gated models
export HF_HUB_OFFLINE=1                  # disable network access (use cache only)
export TRANSFORMERS_OFFLINE=1            # offline mode for transformers

# Python
export TOKENIZERS_PARALLELISM=false    # suppress tokenizer warning
export OMP_NUM_THREADS=8               # OpenMP thread count
```

---

## F.9 Complete nginx Reverse Proxy Configuration

Production-grade nginx config with rate limiting, TLS termination, and vLLM health-check integration:

```nginx
# /etc/nginx/sites-available/vllm-api
# nginx 1.25+ (for http2 directive)

upstream vllm_backend {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=10s;
    # Add more backends for multi-instance:
    # server 127.0.0.1:8001 max_fails=3 fail_timeout=10s;
    # server 127.0.0.1:8002 max_fails=3 fail_timeout=10s;
    keepalive 64;
}

# Rate limiting zones
limit_req_zone $http_authorization zone=per_api_key:20m rate=60r/m;
limit_req_zone $binary_remote_addr zone=per_ip:10m    rate=100r/m;
limit_conn_zone $http_authorization zone=conn_api:20m;

# Log format with latency
log_format vllm_log '$remote_addr - $http_authorization [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    'rt=$request_time uct=$upstream_connect_time '
                    'uht=$upstream_header_time urt=$upstream_response_time';

server {
    listen 443 ssl;
    http2 on;
    server_name api.yourdomain.com;

    # TLS (use cert-manager or Let's Encrypt in production)
    ssl_certificate     /etc/ssl/certs/api.yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/api.yourdomain.com.key;
    ssl_protocols       TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_session_cache   shared:SSL:10m;

    access_log /var/log/nginx/vllm_access.log vllm_log;
    error_log  /var/log/nginx/vllm_error.log warn;

    # Security headers
    add_header X-Content-Type-Options  nosniff;
    add_header X-Frame-Options         SAMEORIGIN;
    add_header Strict-Transport-Security "max-age=31536000" always;

    # Large body for long prompts (128K tokens ≈ 512KB text)
    client_max_body_size 4m;
    client_body_timeout  30s;

    # Inference API
    location /v1/ {
        # Rate limiting
        limit_req zone=per_api_key burst=10 nodelay;
        limit_req zone=per_ip      burst=20 nodelay;
        limit_conn conn_api 5;
        limit_req_status 429;
        add_header Retry-After 1;

        # Proxy settings
        proxy_pass          http://vllm_backend;
        proxy_http_version  1.1;
        proxy_set_header    Connection "";
        proxy_set_header    Host $host;
        proxy_set_header    X-Real-IP $remote_addr;
        proxy_set_header    X-Forwarded-For $proxy_add_x_forwarded_for;

        # Streaming support (SSE)
        proxy_buffering     off;
        proxy_cache         off;
        proxy_read_timeout  300s;   # long timeout for slow completions
        proxy_send_timeout  60s;

        # Pass Authorization header through
        proxy_set_header    Authorization $http_authorization;
    }

    # Health check (no auth, no rate limit)
    location /health {
        proxy_pass http://vllm_backend/health;
        access_log off;
    }

    # Metrics (internal only)
    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        deny  all;
        proxy_pass http://127.0.0.1:8000/metrics;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

---

## F.10 Systemd Service with Auto-Restart and Memory Limits

```ini
# /etc/systemd/system/vllm-serve.service
[Unit]
Description=vLLM OpenAI-Compatible API Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=vllm
Group=vllm
WorkingDirectory=/opt/vllm

# Environment file (keep secrets out of unit file)
EnvironmentFile=/etc/vllm/env

# Command
ExecStart=/opt/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /models/Meta-Llama-3.1-8B-Instruct \
    --served-model-name llama3-8b \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --port 8000

# Restart policy
Restart=on-failure
RestartSec=10s
StartLimitIntervalSec=120s
StartLimitBurst=5

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity     # required for GPU pinned memory

# Graceful shutdown: wait 5min for in-flight requests to complete
TimeoutStopSec=300
KillSignal=SIGTERM
KillMode=mixed

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vllm

[Install]
WantedBy=multi-user.target
```

```bash
# Deploy and enable
sudo systemctl daemon-reload
sudo systemctl enable vllm-serve
sudo systemctl start vllm-serve
sudo systemctl status vllm-serve

# Follow logs
journalctl -u vllm-serve -f

# Graceful restart (waits for in-flight requests)
sudo systemctl reload-or-restart vllm-serve
```

---

## F.11 Production Startup Checklist

Run through this before going live with any new deployment:

```bash
#!/bin/bash
# production_preflight.sh

set -e
VLLM_URL="${1:-http://localhost:8000}"
PASS=0; FAIL=0

check() {
  local name="$1"; shift
  if "$@" &>/dev/null; then
    echo "  ✓ $name"; ((PASS++))
  else
    echo "  ✗ $name — FAILED"; ((FAIL++))
  fi
}

echo "Pre-flight check: $VLLM_URL"
echo "=================================================="

# 1. Service reachable
check "Service responds" curl -sf "$VLLM_URL/health"

# 2. Model list
check "Model listed" \
  bash -c "curl -sf '$VLLM_URL/v1/models' | grep -q 'id'"

# 3. Basic completion
check "Completions endpoint works" \
  bash -c "curl -sf -X POST '$VLLM_URL/v1/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"default\",\"prompt\":\"Hello\",\"max_tokens\":5}' \
    | grep -q 'text'"

# 4. Chat completions
check "Chat endpoint works" \
  bash -c "curl -sf -X POST '$VLLM_URL/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5}' \
    | grep -q 'content'"

# 5. Streaming
check "Streaming works" \
  bash -c "curl -sf -X POST '$VLLM_URL/v1/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"default\",\"prompt\":\"Count:\",\"max_tokens\":10,\"stream\":true}' \
    | grep -q 'data:'"

# 6. Metrics exported
check "Prometheus metrics" curl -sf "$VLLM_URL/metrics" | grep -q "vllm_"

# 7. GPU memory not over 95%
check "GPU memory headroom" \
  bash -c "nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits | \
    awk -F',' '{if(\$1/\$2 < 0.95) exit 0; else exit 1}'"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && echo "✓ READY FOR PRODUCTION" || echo "✗ NOT READY"
exit $FAIL
```

