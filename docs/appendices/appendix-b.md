# Appendix B: Installation and Compilation Guide

> *"The hardest part of inference is not the math — it's getting the right CUDA version, the right driver, and the right Python environment to all agree with each other."*

---

This appendix provides complete, tested installation instructions for vLLM and llama.cpp across all major platforms. Follow the section matching your hardware.

---

## B.1 Prerequisites Checklist

Before installing either engine, verify your environment:

```bash
# 1. Check GPU and driver
nvidia-smi

# Expected output (H100 example):
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2    |
# +-----------------------------------------------------------------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile... |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util   |
# |   0  NVIDIA H100 SXM               On  | 00000000:3B:00.0 Off |          0 |
# | N/A   38C    P0             72W / 700W |   1024MiB / 81920MiB |      0%    |
# +-----------------------------------------------------------------------------+

# 2. Check CUDA toolkit
nvcc --version

# 3. Check Python version (3.9-3.12 supported)
python3 --version

# 4. Check available disk space (vLLM needs ~5GB, models need 14GB+ each)
df -h /
```

### B.1.1 CUDA Compatibility Matrix

```
CUDA Version  | Min Driver (Linux)  | Min Driver (Windows) | vLLM Support
─────────────────────────────────────────────────────────────────────────
12.4          | 550.54.14           | 551.61               | ✓ (preferred)
12.3          | 545.23.06           | 545.84.02            | ✓
12.2          | 535.54.03           | 536.25               | ✓
12.1          | 530.30.02           | 531.14               | ✓ (legacy)
11.8          | 520.61.05           | 522.06               | ✓ (legacy)
< 11.8        | —                   | —                    | ✗

GPU Architecture Support:
  Ampere (A100, A6000, RTX 3090): CUDA 11.1+
  Ada Lovelace (RTX 4090, L40S):  CUDA 11.8+
  Hopper (H100, H200):             CUDA 12.0+
  Blackwell (B100, B200):          CUDA 12.4+
```

---

## B.2 vLLM Installation

### B.2.1 Quick Install (pip)

```bash
# Create a clean virtual environment
python3 -m venv vllm-env
source vllm-env/bin/activate  # Linux/macOS
# .\vllm-env\Scripts\activate   # Windows

# Upgrade pip first (important)
pip install --upgrade pip setuptools wheel

# Install vLLM (CUDA 12.4 wheels)
pip install vllm

# For specific CUDA version:
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

### B.2.2 Installation Verification

```bash
# Test 1: Import check
python -c "import vllm; print(f'vLLM {vllm.__version__} installed')"

# Test 2: GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Test 3: Minimal inference (downloads ~500MB)
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='facebook/opt-125m')  # tiny test model
result = llm.generate(['Hello,'], SamplingParams(max_tokens=10))
print(result[0].outputs[0].text)
"
```

### B.2.3 Docker Installation (Recommended for Production)

```bash
# Pull official vLLM image (check Docker Hub for latest tag)
docker pull vllm/vllm-openai:latest

# Run with GPU passthrough
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16

# With specific GPU (e.g., GPU 0 only)
docker run --runtime nvidia --gpus '"device=0"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct

# Test the server
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello", "max_tokens": 50}'
```

### B.2.4 Building vLLM from Source

Build from source when you need unreleased features or custom modifications:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install build dependencies
pip install -r requirements-build.txt

# Build (takes 10-20 minutes)
pip install -e . --no-build-isolation

# For specific CUDA compute capability (skip auto-detection):
TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" pip install -e . --no-build-isolation
# 8.0 = A100, 8.6 = RTX 3090/A40, 9.0 = H100

# Enable FlashInfer (faster attention kernels):
VLLM_ATTENTION_BACKEND=FLASHINFER pip install -e . --no-build-isolation
```

### B.2.5 Common vLLM Installation Errors

```
ERROR: Could not find a version that satisfies the requirement vllm
  → Your pip is outdated: pip install --upgrade pip

RuntimeError: CUDA error: no kernel image is available for execution
  → CUDA version mismatch. Check nvcc --version vs torch.version.cuda
  → Reinstall: pip install torch --index-url https://download.pytorch.org/whl/cuXXX

ImportError: libcuda.so.1: cannot open shared object file
  → CUDA not on library path: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

OutOfMemoryError during startup
  → Model too large for GPU. Use --dtype float16 or --quantization awq

ValueError: The model's max seq len (X) is larger than the maximum number of tokens
  → Lower --max-model-len to fit KV cache within GPU memory
```

---

## B.3 llama.cpp Installation

### B.3.1 Build from Source (Linux/macOS)

```bash
# Clone repository
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build with CMake (recommended)
cmake -B build \
    -DGGML_CUDA=ON \          # Enable CUDA support
    -DCMAKE_CUDA_ARCHITECTURES=native  # Auto-detect GPU arch

cmake --build build --config Release -j$(nproc)

# Binaries are in ./build/bin/
ls build/bin/
# llama-cli  llama-server  llama-bench  llama-perplexity  llama-embedding
```

### B.3.2 Build Flags Reference

```bash
# CPU-only (no GPU)
cmake -B build
cmake --build build --config Release -j$(nproc)

# CUDA (NVIDIA GPU)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# CUDA with specific architecture
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
cmake --build build --config Release -j$(nproc)

# Metal (Apple Silicon)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(nproc)

# Vulkan (AMD/Intel GPU, cross-platform)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

# OpenCL (legacy AMD GPU)
cmake -B build -DGGML_OPENCL=ON
cmake --build build --config Release -j$(nproc)

# ROCm (AMD GPU, Linux only)
cmake -B build -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="gfx1100;gfx1030"  # adjust to your GPU
cmake --build build --config Release -j$(nproc)
```

### B.3.3 Windows Build

```powershell
# Install prerequisites: Visual Studio 2022, CUDA Toolkit, CMake

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Configure
cmake -B build -DGGML_CUDA=ON -G "Visual Studio 17 2022" -A x64

# Build
cmake --build build --config Release

# Binaries in .\build\bin\Release\
```

### B.3.4 Verification

```bash
# Download a test model (Qwen2.5-0.5B-Instruct GGUF, ~300MB)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    --include "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
    --local-dir ./models

# Run inference test
./build/bin/llama-cli \
    -m ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -p "Hello, world!" \
    -n 20 \
    --no-mmap

# Expected output: some text generated, then statistics like:
# llama_print_timings: load time = 123.45 ms
# llama_print_timings:   eval time = 456.78 ms / 20 runs (22.84 ms per token)
```

### B.3.5 Common llama.cpp Build Errors

```
CUDA-related:
  nvcc not found
    → export PATH=/usr/local/cuda/bin:$PATH

  No CUDA-capable device detected
    → nvidia-smi works but llama.cpp uses CUDA: rebuild with -DGGML_CUDA=ON

  Segfault on model load
    → Corrupted GGUF file. Re-download model.

Memory-related:
  GGML_ASSERT: n_ctx * n_batch * n_ubatch <= 32768
    → Reduce context (-c 4096) or batch size

General build:
  CMake version too old (< 3.14)
    → brew upgrade cmake  or  pip install cmake
```

---

## B.4 Model Download

### B.4.1 Hugging Face CLI

```bash
# Install
pip install huggingface_hub

# Authenticate (required for gated models like Llama)
huggingface-cli login
# Enter your HF token from https://huggingface.co/settings/tokens

# Download full model (for vLLM)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ./models/llama-3.1-8b

# Download specific file (for llama.cpp GGUF)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
    --local-dir ./models

# Download with progress bar and resume capability
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
    --local-dir ./models/llama-3.1-70b \
    --resume-download
```

### B.4.2 Python Download

```python
from huggingface_hub import snapshot_download, hf_hub_download

# Download complete model snapshot
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="./models/qwen2.5-7b",
    ignore_patterns=["*.msgpack", "*.h5"],  # skip TF/Flax weights
)

# Download single file
hf_hub_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q4_k_m.gguf",
    local_dir="./models",
)
```

### B.4.3 Model Size Reference

```
Model                    | Format | Download Size | VRAM Needed
──────────────────────────────────────────────────────────────
Qwen2.5-0.5B-Instruct   | BF16   |    1.0 GB     |  2 GB
Qwen2.5-7B-Instruct     | BF16   |   14.5 GB     | 16 GB
Qwen2.5-7B-Instruct     | Q4_K_M |    4.4 GB     |  6 GB
Llama-3.1-8B-Instruct   | BF16   |   16.0 GB     | 18 GB
Llama-3.1-8B-Instruct   | Q4_K_M |    4.7 GB     |  6 GB
Qwen2.5-14B-Instruct    | BF16   |   29.0 GB     | 32 GB
Qwen2.5-32B-Instruct    | Q4_K_M |   18.5 GB     | 22 GB
Llama-3.1-70B-Instruct  | BF16   |  140.0 GB     | 160 GB (2×H100)
Llama-3.1-70B-Instruct  | Q4_K_M |   40.0 GB     | 48 GB
Qwen2.5-72B-Instruct    | GPTQ-4 |   38.0 GB     | 44 GB
DeepSeek-V3             | FP8    |  ~350 GB      | 8×H100 recommended
```

---

## B.5 Multi-GPU Setup

### B.5.1 NCCL Configuration

vLLM uses NCCL for tensor parallelism. Key environment variables:

```bash
# Disable IB (InfiniBand) if not available
export NCCL_IB_DISABLE=1

# Use P2P communication between GPUs (faster for NVLink)
export NCCL_P2P_LEVEL=NVL

# For multi-node (use appropriate network interface)
export NCCL_SOCKET_IFNAME=eth0

# Debug NCCL issues
export NCCL_DEBUG=INFO

# Launch vLLM with tensor parallelism
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16
```

### B.5.2 Checking NVLink

```bash
# Check NVLink connectivity between GPUs
nvidia-smi nvlink --status

# Check P2P access between GPU 0 and GPU 1
python3 -c "
import torch
torch.cuda.set_device(0)
print(f'P2P 0→1: {torch.cuda.can_device_access_peer(0, 1)}')
print(f'P2P 1→0: {torch.cuda.can_device_access_peer(1, 0)}')
"
```

---

## B.6 Environment Management

### B.6.1 conda Environment

```bash
# Create environment with specific Python + CUDA
conda create -n vllm python=3.11 cuda-toolkit=12.4 -c nvidia
conda activate vllm

# Install vLLM
pip install vllm

# Save environment
conda env export > vllm-environment.yml

# Recreate on another machine
conda env create -f vllm-environment.yml
```

### B.6.2 Docker Compose for Production

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --dtype bfloat16
      --max-model-len 32768
      --gpu-memory-utilization 0.85
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## B.7 SGLang Installation

SGLang is an alternative serving framework with strong structured generation support:

```bash
# Install SGLang
pip install "sglang[all]"

# Start server
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --tp 1

# Test
python -c "
import sglang as sgl

@sgl.function
def chain_of_thought(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen('answer', max_tokens=256))

state = chain_of_thought.run(question='What is 15 × 23?')
print(state['answer'])
"
```

---

## B.8 TensorRT-LLM Installation

TensorRT-LLM requires specific NVIDIA toolchain versions:

```bash
# Prerequisites: CUDA 12.x, cuDNN 8.x, TensorRT 9.x
# Best approach: use official NGC container

# Pull TRT-LLM container (update tag to latest)
docker pull nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3

# Or install directly (may have dependency conflicts)
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

# Verify
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# Convert and build engine (Llama-3.1-8B example)
python convert_checkpoint.py \
    --model_dir /path/to/llama-3.1-8b \
    --output_dir ./trt_ckpt/llama-8b \
    --dtype float16

trtllm-build \
    --checkpoint_dir ./trt_ckpt/llama-8b \
    --output_dir ./trt_engines/llama-8b \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 512
```

---

## B.9 Troubleshooting Quick Reference

```
Symptom                             | Likely Cause          | Fix
────────────────────────────────────────────────────────────────────────────
ImportError: vllm                   | Not installed         | pip install vllm
CUDA out of memory                  | Model too large       | Use quantization or smaller model
Slow first request (60s+)           | Model loading         | Expected; warm up with dummy request
Tokens/s much lower than expected   | Batch size too small  | Increase --max-num-seqs
vllm serve hangs at startup         | NCCL init             | export NCCL_IB_DISABLE=1
llama.cpp: illegal instruction      | Wrong arch binary     | Rebuild with -DCMAKE_CUDA_ARCHITECTURES=native
Model produces garbage              | Wrong chat template   | Use --chat-template correctly
OOM during prefill but not decode   | Context too long      | Reduce --max-model-len
404 /v1/completions                 | Wrong endpoint        | vLLM uses /v1/completions and /v1/chat/completions
```

---

## B.10 Version Compatibility Table

```
vLLM Version | PyTorch | CUDA  | Python | Notes
─────────────────────────────────────────────────────────────
0.6.x        | 2.5.x   | 12.4  | 3.9-3.12 | Current stable
0.5.x        | 2.4.x   | 12.2  | 3.8-3.12 | Previous stable  
0.4.x        | 2.3.x   | 12.1  | 3.8-3.11 | Legacy

llama.cpp    | GGUF    | Build System  | Notes
────────────────────────────────────────────────────────
b4000+       | v3      | CMake         | Current (2025+)
b3000-b3999  | v3      | CMake/Makefile| Transition period
< b3000      | v2/v3   | Makefile      | Legacy (avoid)
```

---

## B.11 Docker Installation (Recommended for Production)

Running vLLM and llama.cpp in Docker ensures reproducible environments and simplifies deployment. These are pinned to specific versions — update the tags when upgrading.

### vLLM Docker (CUDA 12.4, PyTorch 2.5)

```dockerfile
# Dockerfile.vllm
FROM vllm/vllm-openai:v0.6.3

# Pin Python deps for reproducibility
RUN pip install --no-cache-dir \
    openai==1.54.0 \
    prometheus-client==0.21.0 \
    pyzmq==26.2.0

# Copy model weights (or mount at runtime)
# COPY models/ /models/

# Environment
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV TOKENIZERS_PARALLELISM=false
ENV NCCL_ASYNC_ERROR_HANDLING=1

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
```

```bash
# Build and run vLLM container (single H100)
docker build -t my-vllm:v0.6.3 -f Dockerfile.vllm .

docker run -d \
  --name vllm-serve \
  --gpus '"device=0"' \
  --ipc=host \
  --network=host \
  -v /data/models:/models \
  -v /tmp/vllm-cache:/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN}" \
  my-vllm:v0.6.3 \
    --model /models/Meta-Llama-3.1-8B-Instruct \
    --served-model-name llama3-8b \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --dtype bfloat16 \
    --port 8000
```

### Multi-GPU vLLM with Docker Compose

```yaml
# docker-compose.yml — 4-GPU tensor parallel deployment
version: "3.9"

services:
  vllm:
    image: vllm/vllm-openai:v0.6.3
    ipc: host
    network_mode: host
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            count: 4
            capabilities: [gpu]
    volumes:
      - /data/models:/models:ro
      - /data/hf-cache:/root/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - NCCL_ASYNC_ERROR_HANDLING=1
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
    command: >
      --model /models/Meta-Llama-3.1-70B-Instruct
      --tensor-parallel-size 4
      --max-model-len 16384
      --max-num-seqs 512
      --dtype bfloat16
      --gpu-memory-utilization 0.92
      --enable-prefix-caching
      --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
    restart: unless-stopped
```

### llama.cpp Docker

```dockerfile
# Dockerfile.llama-cpp
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git cmake build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Pin llama.cpp to a specific build tag
ARG LLAMA_CPP_TAG=b4235
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} \
    https://github.com/ggerganov/llama.cpp.git /llama.cpp

WORKDIR /llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;89;90" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc) \
    && cp build/bin/llama-server /usr/local/bin/

EXPOSE 8080

ENTRYPOINT ["llama-server"]
```

```bash
# Run llama.cpp server
docker build -t my-llamacpp:b4235 -f Dockerfile.llama-cpp .

docker run -d \
  --name llamacpp-serve \
  --gpus '"device=0"' \
  -p 8080:8080 \
  -v /data/models:/models:ro \
  my-llamacpp:b4235 \
    --model /models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 8192 \
    --parallel 8 \
    --host 0.0.0.0 \
    --port 8080
```

---

## B.12 Conda Environment (Alternative to Docker)

For development machines where Docker overhead is undesirable:

```bash
# Create pinned environment
conda create -n vllm-dev python=3.11 -y
conda activate vllm-dev

# Install CUDA toolkit matching your driver
conda install -c nvidia cuda-toolkit=12.4 -y

# Install PyTorch (pinned)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install vLLM (pinned)
pip install vllm==0.6.3

# Install llama-cpp-python with CUDA
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
pip install llama-cpp-python==0.3.2 --no-cache-dir

# Save exact environment
pip freeze > requirements-vllm-dev.txt
conda env export > environment-vllm-dev.yml
```

---

## B.13 Automated Health Check Script

Run after any installation to verify everything works end-to-end:

```bash
#!/bin/bash
# health_check.sh — verify vLLM and llama.cpp installations

set -e
GREEN='\033[0;32m' RED='\033[0;31m' NC='\033[0m'

check() {
  if eval "$2" &>/dev/null; then
    echo -e "${GREEN}✓${NC} $1"
  else
    echo -e "${RED}✗${NC} $1 — FAILED"
    eval "$2" 2>&1 | head -5
  fi
}

echo "=== vLLM Health Check ==="
check "Python 3.9+"          "python3 --version | grep -E '3\.(9|10|11|12)'"
check "CUDA available"       "python3 -c 'import torch; assert torch.cuda.is_available()'"
check "vLLM importable"      "python3 -c 'import vllm; print(vllm.__version__)'"
check "vLLM serve starts"    "timeout 5 python3 -m vllm.entrypoints.openai.api_server --help"

echo ""
echo "=== llama.cpp Health Check ==="
check "llama-server binary"  "which llama-server || ls build/bin/llama-server"
check "CUDA backend"         "llama-server --version 2>&1 | grep -i cuda"

echo ""
echo "=== GPU Check ==="
check "nvidia-smi"           "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
check "GPU memory (>20GB)"   "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{if(\$1>20000) exit 0; else exit 1}'"

echo ""
echo "Installation check complete."
```



---

## B.14 Cloud GPU Provisioning

Running vLLM or llama.cpp locally requires a capable GPU. When you are working on a laptop, a build server, or a machine without a discrete NVIDIA GPU, a cloud instance is the fastest path to a working environment. This section walks through five platforms — AWS EC2, Lambda Labs, RunPod, Modal, and Vast.ai — with enough detail to go from zero to a running vLLM server on each.

### B.14.1 Platform Comparison at a Glance

```
Platform     GPU Selection    Pricing Model     Best For
──────────────────────────────────────────────────────────────────────────
AWS EC2      Wide (A100/H100) On-demand/Spot    Production, compliance, VPC
Lambda Labs  A100/H100/A10G   Hourly on-demand  Research, simple setup
RunPod       A100/H100/RTX    Hourly pods       Dev/experimentation, cheap
Modal        A100/H100/T4     Per-second billed Serverless functions, CI
Vast.ai      Mixed (market)   Bid/on-demand     Cheapest H100, flexible
──────────────────────────────────────────────────────────────────────────

Approximate spot prices for one H100 SXM5 80GB (May 2026, varies):
  AWS EC2 p5.xlarge:    $6.98/hr on-demand,  ~$2.50/hr spot
  Lambda Labs H100:     $2.99/hr on-demand   (no spot)
  RunPod H100 SXM:      $2.49/hr on-demand   (community cloud)
  Modal H100:           $4.63/hr (billed per second)
  Vast.ai H100 SXM:     $1.89–$3.20/hr       (marketplace bid)
```

---

### B.14.2 AWS EC2

AWS EC2 is the right choice when you need production-grade SLAs, VPC networking, IAM roles, compliance certifications, or tight integration with S3, EFS, or ECR.

**Recommended instance types for LLM inference:**

```
Instance         GPU                  VRAM     Use case
───────────────────────────────────────────────────────────────────
g4dn.xlarge      T4 (1×)              16 GB    7B Q4 models, dev/test
g4dn.12xlarge    T4 (4×)              64 GB    13B–34B models
g5.xlarge        A10G (1×)            24 GB    7B BF16, 13B Q4
g5.12xlarge      A10G (4×)            96 GB    70B Q4, 34B BF16
p3.2xlarge       V100 (1×)            16 GB    Legacy; prefer g5
p4d.24xlarge     A100 40GB (8×)       320 GB   70B BF16, large batches
p4de.24xlarge    A100 80GB (8×)       640 GB   2×70B, MoE models
p5.48xlarge      H100 SXM5 (8×)       640 GB   Frontier models
```

**Step 1 — Launch via AWS CLI:**

```bash
# Install and configure CLI (if not done already)
pip install awscli
aws configure   # enter Access Key, Secret Key, region (e.g. us-east-1)

# Find the latest Deep Learning AMI (Ubuntu 22.04, CUDA 12.x pre-installed)
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*" \
  --query "sort_by(Images, &CreationDate)[-1].[ImageId,Name]" \
  --output text
# e.g. ami-0abcd1234efgh5678  Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 ...

# Create a key pair (skip if you have one)
aws ec2 create-key-pair --key-name my-gpu-key \
  --query 'KeyMaterial' --output text > ~/.ssh/my-gpu-key.pem
chmod 600 ~/.ssh/my-gpu-key.pem

# Launch a g5.xlarge (cheapest CUDA instance with 24 GB VRAM)
aws ec2 run-instances \
  --image-id ami-0abcd1234efgh5678 \
  --instance-type g5.xlarge \
  --key-name my-gpu-key \
  --security-group-ids sg-xxxxxxxxxxxxxxxxx \
  --subnet-id subnet-xxxxxxxxxxxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vllm-dev}]' \
  --query 'Instances[0].InstanceId' \
  --output text
# Returns: i-0123456789abcdef0

# Wait until running
aws ec2 wait instance-running --instance-ids i-0123456789abcdef0

# Get public IP
aws ec2 describe-instances \
  --instance-ids i-0123456789abcdef0 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

**Step 2 — Launch via AWS Console (alternative):**

Navigate to EC2 → Launch Instance → choose "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" (search in Community AMIs) → select instance type → configure storage (200 GB minimum for models) → launch.

**Step 3 — SSH and install vLLM:**

```bash
# SSH in
ssh -i ~/.ssh/my-gpu-key.pem ubuntu@<PUBLIC_IP>

# The Deep Learning AMI has CUDA and conda pre-installed
# Activate the PyTorch environment
conda activate pytorch  # or check: conda env list

# Verify GPU
nvidia-smi

# Install vLLM
pip install vllm

# Start serving (port 8000 must be open in the security group)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000
```

**Step 4 — Use Spot Instances for 60–75% cost reduction:**

```bash
# Request a spot instance (interruption risk, save money for dev work)
aws ec2 run-instances \
  --image-id ami-0abcd1234efgh5678 \
  --instance-type g5.xlarge \
  --key-name my-gpu-key \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.80","SpotInstanceType":"one-time"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]'

# Check current spot prices
aws ec2 describe-spot-price-history \
  --instance-types g5.xlarge p4d.24xlarge \
  --product-descriptions "Linux/UNIX" \
  --start-time $(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --query 'SpotPriceHistory[*].[InstanceType,SpotPrice,Timestamp]' \
  --output table
```

**EC2-specific tips:**

- Always attach an **IAM instance profile** if the instance needs to access S3 model buckets — avoid hardcoding credentials.
- Open only port 22 (SSH) and 8000 (vLLM) in the security group; restrict to your IP: `--cidr <YOUR_IP>/32`.
- Use **EFS** (Elastic File System) to share model weights across multiple instances without re-downloading.
- For multi-GPU tensor parallelism across instances, use the **p4d/p5 cluster placement group** with EFA networking (`--placement-group my-cluster-pg`).
- **Stop** (not terminate) an instance to preserve the root volume between sessions; you pay only for EBS storage while stopped (~$0.08/GB/month for gp3).

---

### B.14.3 Lambda Labs

Lambda Labs offers GPU cloud instances with a simpler interface than AWS and competitive on-demand pricing. No spot instances, but no bidding complexity either. Ideal for researchers and individuals who want a clean Ubuntu environment with CUDA already configured.

**Available GPU instances (representative, check current availability):**

```
Instance Name        GPU              VRAM     Price/hr
──────────────────────────────────────────────────────
gpu_1x_a10           A10 (1×)         24 GB    $0.60
gpu_1x_a100_sxm4     A100 SXM4 (1×)  40 GB    $1.29
gpu_1x_h100_sxm5     H100 SXM5 (1×)  80 GB    $2.99
gpu_8x_h100_sxm5     H100 SXM5 (8×)  640 GB   $23.92
gpu_1x_a100_80gb     A100 (1×)       80 GB    $1.99
```

**Step 1 — Create an instance via the Lambda Cloud API:**

```bash
# Install Lambda CLI (or use the web console at cloud.lambdalabs.com)
pip install lambda-cloud

# Set API key (get from cloud.lambdalabs.com/api-keys)
export LAMBDA_API_KEY="your_api_key_here"

# List available instance types
curl -u "${LAMBDA_API_KEY}:" \
  https://cloud.lambdalabs.com/api/v1/instance-types \
  | python3 -m json.tool

# List available regions for H100
curl -u "${LAMBDA_API_KEY}:" \
  "https://cloud.lambdalabs.com/api/v1/instance-types" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)['data']
for name, info in data.items():
    if 'h100' in name:
        regions = list(info.get('regions_with_capacity_available', {}).keys())
        print(f'{name}: {regions}')
"

# Add your SSH key first (one-time)
curl -u "${LAMBDA_API_KEY}:" \
  -X POST https://cloud.lambdalabs.com/api/v1/ssh-keys \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"my-key\", \"public_key\": \"$(cat ~/.ssh/id_rsa.pub)\"}"
# Returns: {"data": {"id": "abc123", "name": "my-key", ...}}

# Launch an H100 instance
curl -u "${LAMBDA_API_KEY}:" \
  -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/launch \
  -H "Content-Type: application/json" \
  -d '{
    "region_name": "us-west-2",
    "instance_type_name": "gpu_1x_h100_sxm5",
    "ssh_key_names": ["my-key"],
    "file_system_names": [],
    "quantity": 1,
    "name": "vllm-h100"
  }' | python3 -m json.tool
# Returns instance id and IP

# List running instances
curl -u "${LAMBDA_API_KEY}:" \
  https://cloud.lambdalabs.com/api/v1/instances \
  | python3 -m json.tool
```

**Step 2 — SSH and set up vLLM:**

```bash
# Lambda instances use 'ubuntu' user, CUDA 12.x pre-installed
ssh ubuntu@<INSTANCE_IP>

# Check environment — Lambda AMI has conda and CUDA ready
nvidia-smi
python3 --version   # typically 3.10 or 3.11

# Install vLLM
pip install vllm

# Set HuggingFace token for gated models
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"

# Serve Llama-3.1-8B
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 &

# Test from your local machine
curl http://<INSTANCE_IP>:8000/v1/models
```

**Step 3 — Terminate when done:**

```bash
# Get instance ID from the list command above
curl -u "${LAMBDA_API_KEY}:" \
  -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -H "Content-Type: application/json" \
  -d '{"instance_ids": ["abc123defg456"]}'
```

**Lambda-specific tips:**

- Lambda instances have **no persistent storage by default** — attach a Lambda filesystem (NFS-backed, $0.20/GB/month) to preserve model weights between sessions.
- Lambda does not charge for stopped instances — there is no "stop" state; you pay while running, nothing when terminated.
- Port 8000 is **open by default** on Lambda instances (unlike AWS where you configure security groups). Anyone who knows the IP can reach your vLLM server — use `--api-key` authentication or an SSH tunnel for anything sensitive.
- Lambda **availability is limited** — H100 instances sell out. Check the console or use the API to poll `regions_with_capacity_available` before building automation around specific instance types.

---

### B.14.4 RunPod

RunPod offers GPU pods with per-minute billing, a marketplace of community cloud GPUs, and a clean web UI. It is the most flexible in terms of GPU variety (including older A6000, 3090, 4090 for small models) and often the cheapest option for H100s in the community cloud tier.

**Two tiers:**

- **Secure Cloud**: RunPod-operated data centres, higher reliability, slightly higher price.
- **Community Cloud**: Third-party hosts, lower price, occasional interruptions.

**Step 1 — Launch a pod via RunPod CLI:**

```bash
# Install RunPod CLI
pip install runpod

# Set API key (from runpod.io/console/user/settings)
export RUNPOD_API_KEY="your_runpod_api_key"
runpod config  # or set env var

# List available GPU types
runpod gpu list

# Programmatic launch via Python SDK
python3 - <<'EOF'
import runpod

runpod.api_key = "your_runpod_api_key"

pod = runpod.create_pod(
    name="vllm-h100",
    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    gpu_type_id="NVIDIA H100 80GB HBM3",  # exact string from gpu list
    cloud_type="SECURE",   # or "COMMUNITY" for cheaper
    gpu_count=1,
    volume_in_gb=100,       # persistent storage
    container_disk_in_gb=50,
    ports="8000/http,22/tcp",
    env={
        "HF_TOKEN": "hf_xxxx",
    },
)
print(f"Pod ID: {pod['id']}")
print(f"Status: {pod['desiredStatus']}")
EOF
```

**Step 2 — Connect and install vLLM:**

```bash
# SSH via RunPod's proxy (shown in the web console)
# Or use the web terminal at runpod.io/console/pods

# RunPod Docker image has CUDA but may not have vLLM pre-installed
pip install vllm

# Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000

# Access via RunPod's proxy URL (shown in console):
# https://<POD_ID>-8000.proxy.runpod.net/v1/models
```

**Step 3 — Use a pre-built vLLM template (fastest start):**

RunPod provides community templates. Use the vLLM template to skip manual installation:

```python
import runpod

pod = runpod.create_pod(
    name="vllm-ready",
    # Pre-built vLLM image from RunPod's template library
    image_name="vllm/vllm-openai:latest",
    gpu_type_id="NVIDIA H100 80GB HBM3",
    cloud_type="COMMUNITY",
    gpu_count=1,
    volume_in_gb=200,
    container_disk_in_gb=20,
    ports="8000/http",
    docker_args=(
        "--model meta-llama/Llama-3.1-8B-Instruct "
        "--dtype bfloat16 "
        "--host 0.0.0.0 "
        "--port 8000"
    ),
    env={"HF_TOKEN": "hf_xxxx"},
)
```

**Step 4 — Terminate the pod:**

```python
import runpod
runpod.api_key = "your_api_key"
runpod.terminate_pod("your_pod_id")
```

**RunPod-specific tips:**

- Use **volume storage** (`volume_in_gb`) for model weights — it persists across pod terminations and re-attaches. Container disk is wiped when the pod is deleted.
- RunPod's **proxy URLs** (`<POD_ID>-8000.proxy.runpod.net`) let you reach your vLLM server without configuring firewall rules, but they add ~10–30ms latency. For performance testing, use direct IP access via SSH port forwarding instead.
- **Spot pods** on RunPod (called "interruptible") are not yet standard — check the console for current options.
- Community cloud GPUs vary in quality. If you get poor throughput, stop and relaunch to land on a different host.
- RunPod bills per **minute**, not per hour — good for short experiments where you spin up, run a benchmark, and terminate within 30 minutes.

---

### B.14.5 Modal

Modal is a serverless GPU platform — you write Python functions decorated with `@app.function(gpu=...)` and Modal handles provisioning, scaling, and teardown automatically. There are no long-running instances to manage. It is the right choice for batch inference pipelines, CI evaluation harnesses, or any workload that is event-driven rather than always-on.

**Step 1 — Install and authenticate:**

```bash
pip install modal
modal setup   # opens browser for auth (GitHub / Google)
```

**Step 2 — Write a Modal function for vLLM serving:**

```python
# modal_vllm.py
import modal

# Define the container image with vLLM pre-installed
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.3", "huggingface_hub")
    .env({"HF_TOKEN": modal.Secret.from_name("hf-token")})
)

app = modal.App("vllm-server")

# Cache the model weights in a Modal Volume (persists across runs)
model_volume = modal.Volume.from_name("model-weights", create_if_missing=True)
MODEL_DIR = "/models"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


@app.function(
    image=vllm_image,
    gpu="H100",                  # or "A100", "A10G", "T4"
    volumes={MODEL_DIR: model_volume},
    timeout=3600,                # 1 hour max runtime
    allow_concurrent_inputs=100, # handle 100 concurrent requests
)
@modal.web_endpoint(method="POST")
def generate(request: dict) -> dict:
    """Stateless inference endpoint — Modal starts the vLLM engine per container."""
    import os
    from vllm import LLM, SamplingParams

    # Engine is initialised once per container (Modal reuses warm containers)
    if not hasattr(generate, "_llm"):
        generate._llm = LLM(
            model=os.path.join(MODEL_DIR, MODEL_NAME.split("/")[-1]),
            dtype="bfloat16",
            max_model_len=8192,
        )

    prompt = request.get("prompt", "Hello")
    max_tokens = request.get("max_tokens", 100)
    params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    outputs = generate._llm.generate([prompt], params)
    return {"text": outputs[0].outputs[0].text}


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={MODEL_DIR: model_volume},
    timeout=600,
)
def download_model():
    """Run once to cache the model in the Modal Volume."""
    from huggingface_hub import snapshot_download
    import os
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=os.path.join(MODEL_DIR, MODEL_NAME.split("/")[-1]),
        ignore_patterns=["*.msgpack", "*.h5"],
    )
    model_volume.commit()
    print(f"Model downloaded to {MODEL_DIR}")


if __name__ == "__main__":
    with app.run():
        download_model.remote()
```

**Step 3 — Deploy and call:**

```bash
# Download the model into the Volume (one-time, ~5 min)
modal run modal_vllm.py::download_model

# Deploy the endpoint (returns a URL)
modal deploy modal_vllm.py
# Deployed: https://your-workspace--vllm-server-generate.modal.run

# Call the endpoint
curl -X POST https://your-workspace--vllm-server-generate.modal.run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain attention in one paragraph.", "max_tokens": 150}'
```

**Step 4 — Batch inference (Modal's strongest use case):**

```python
# batch_inference.py — score 10,000 prompts in parallel on Modal
import modal

app = modal.App("batch-score")

@app.function(
    image=modal.Image.debian_slim().pip_install("vllm"),
    gpu="A10G",
    concurrency_limit=20,          # up to 20 parallel GPU containers
    volumes={"/models": modal.Volume.from_name("model-weights")},
    timeout=600,
)
def score_batch(prompts: list[str]) -> list[str]:
    from vllm import LLM, SamplingParams
    llm = LLM(model="/models/Llama-3.1-8B-Instruct", dtype="bfloat16")
    params = SamplingParams(max_tokens=50, temperature=0.0)
    return [o.outputs[0].text for o in llm.generate(prompts, params)]


@app.local_entrypoint()
def main():
    prompts = [f"Question {i}: what is {i}+{i}?" for i in range(1000)]
    # Split into 50-prompt chunks and fan out across 20 containers
    chunk_size = 50
    chunks = [prompts[i:i+chunk_size] for i in range(0, len(prompts), chunk_size)]
    # starmap runs all chunks in parallel
    results = list(score_batch.starmap([[c] for c in chunks]))
    flat = [r for batch in results for r in batch]
    print(f"Scored {len(flat)} prompts")
```

```bash
modal run batch_inference.py
```

**Modal-specific tips:**

- Modal bills **per second of GPU time** — you pay for exactly the compute used, with no idle time. A batch job that runs for 3 minutes on an H100 costs $4.63/hr × 3/60 hr ≈ $0.23.
- Use `modal.Volume` to cache model weights; without it, the container re-downloads the model on every cold start (~5 min for an 8B model).
- Cold start for a new container (with a warm Volume) is typically 30–90 seconds for vLLM. Use `keep_warm=1` on the function decorator if you need sub-second latency.
- Modal's `concurrency_limit` controls how many GPU containers it will spin up simultaneously — use this to control maximum spend on parallel jobs.
- For an always-on server (more like RunPod/Lambda), use `@modal.web_endpoint` with `keep_warm=1` and a small `concurrency_limit`.

---

### B.14.6 Vast.ai

Vast.ai is a peer-to-peer GPU marketplace — individual hosts rent out their hardware at prices they set, and you bid or rent on-demand. It offers the widest GPU variety and often the lowest prices for H100s, but reliability varies by host. Best suited for cost-sensitive experimentation and batch workloads where interruption is acceptable.

**Step 1 — Search and rent via the CLI:**

```bash
# Install vastai CLI
pip install vastai

# Set API key (from vast.ai/console > Account > API Key)
vastai set api-key YOUR_API_KEY

# Search for H100 instances on-demand
# Format: vastai search offers [filters] [sort]
vastai search offers \
  'gpu_name=H100_SXM5 num_gpus=1 inet_down>500 reliability>0.95' \
  --order-by 'dph_total asc' \
  --limit 10

# Example output:
# ID       CUDA  CPU  RAM    DISK   VRAM   $/hr  Reliability  Host
# 8123456  12.4  32   128G   500G   80G    1.89  0.98         fast-host-1
# 8234567  12.4  16   64G    200G   80G    2.10  0.97         another-host

# Rent the cheapest available (use the ID from search results)
vastai create instance 8123456 \
  --image "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel" \
  --disk 100 \
  --onstart "pip install vllm && vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000" \
  --env "HF_TOKEN=hf_xxxx" \
  --ssh \
  --open-ports 8000

# List your running instances
vastai show instances
```

**Step 2 — Connect via SSH:**

```bash
# Vast.ai provides the SSH command directly
vastai ssh-url <INSTANCE_ID>
# Returns: ssh -p 12345 root@ssh.vast.ai  (or similar)

# SSH in
ssh -p 12345 root@ssh.vast.ai

# Check GPU and start serving if --onstart didn't run yet
nvidia-smi
pip install vllm  # if not already installed by onstart
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "your-secret-key"  # important: lock down the server
```

**Step 3 — Use a pre-built Docker image (recommended):**

```bash
# Use vLLM's official Docker image to avoid installation time
vastai create instance 8123456 \
  --image "vllm/vllm-openai:latest" \
  --disk 200 \
  --args "--model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --host 0.0.0.0 --port 8000" \
  --env "HF_TOKEN=hf_xxxx" \
  --open-ports 8000 \
  --ssh
```

**Step 4 — Programmatic search and rent with Python:**

```python
import subprocess
import json
import os

def find_cheapest_h100(max_price_per_hr: float = 2.50) -> dict | None:
    """Find the cheapest reliable H100 on Vast.ai."""
    result = subprocess.run([
        "vastai", "search", "offers",
        f"gpu_name=H100_SXM5 num_gpus=1 "
        f"dph_total<{max_price_per_hr} reliability>0.95 "
        f"inet_down>200 disk_space>150",
        "--order-by", "dph_total asc",
        "--limit", "5",
        "--raw",
    ], capture_output=True, text=True)
    offers = json.loads(result.stdout)
    if not offers:
        return None
    return offers[0]  # cheapest


def rent_instance(offer_id: int, hf_token: str) -> str:
    """Rent the instance and return the instance ID."""
    result = subprocess.run([
        "vastai", "create", "instance", str(offer_id),
        "--image", "vllm/vllm-openai:latest",
        "--disk", "200",
        "--args", (
            "--model meta-llama/Llama-3.1-8B-Instruct "
            "--dtype bfloat16 --host 0.0.0.0 --port 8000"
        ),
        "--env", f"HF_TOKEN={hf_token}",
        "--open-ports", "8000",
        "--ssh",
        "--raw",
    ], capture_output=True, text=True)
    response = json.loads(result.stdout)
    return response["new_contract"]


def destroy_instance(instance_id: str):
    subprocess.run(["vastai", "destroy", "instance", instance_id])
    print(f"Instance {instance_id} destroyed.")


if __name__ == "__main__":
    HF_TOKEN = os.environ["HF_TOKEN"]
    offer = find_cheapest_h100(max_price_per_hr=2.50)
    if offer:
        print(f"Found H100 at ${offer['dph_total']:.2f}/hr (host reliability: {offer['reliability']:.2f})")
        instance_id = rent_instance(offer["id"], HF_TOKEN)
        print(f"Instance rented: {instance_id}")
        # ... do your work ...
        # destroy_instance(instance_id)
    else:
        print("No H100 under $2.50/hr with >95% reliability currently available.")
```

**Vast.ai-specific tips:**

- Always filter on `reliability > 0.95` — hosts below this threshold have a material chance of dropping your instance mid-run.
- Filter `inet_down > 200` (Mbps) to ensure model downloads from HuggingFace complete in reasonable time.
- Vast.ai instances are **not isolated** — the host machine runs your container. Never put secrets (API keys, private model weights) on Vast.ai that you cannot afford to expose to the host operator. Use `--api-key` on vLLM to prevent unauthorised access via the open port.
- The `--onstart` script runs asynchronously — allow 3–5 minutes after instance creation before the vLLM server is ready.
- Unlike Lambda/RunPod, **Vast.ai has no built-in volume**. Use `--disk` to allocate local container storage; it is wiped on termination. For persistence, upload/download models from S3 or HuggingFace Hub at the start and end of each session.

---

### B.14.7 Cross-Platform Workflow: Model Download to Serve in One Script

The following script runs identically on all five platforms after SSH access. Copy it to your instance and execute:

```bash
#!/bin/bash
# gpu_setup.sh — universal vLLM setup script for any cloud GPU instance
# Usage: bash gpu_setup.sh <MODEL_ID> [PORT]
# Example: bash gpu_setup.sh meta-llama/Llama-3.1-8B-Instruct 8000

set -euo pipefail
MODEL_ID="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${2:-8000}"

echo "=== GPU Setup Script ==="
echo "Model: ${MODEL_ID}"
echo "Port:  ${PORT}"
echo ""

# 1. Verify GPU
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 2. Ensure Python 3.10+
PYTHON=$(which python3.11 2>/dev/null || which python3.10 2>/dev/null || which python3)
echo "Using Python: $($PYTHON --version)"

# 3. Install vLLM (idempotent)
echo "--- Installing vLLM ---"
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet vllm
echo "vLLM: $($PYTHON -c 'import vllm; print(vllm.__version__)')"

# 4. Set HuggingFace token if provided
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "--- HuggingFace token set ---"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# 5. Determine dtype based on VRAM
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if (( VRAM_MB >= 79000 )); then
  DTYPE="bfloat16"
  MAX_LEN=32768
elif (( VRAM_MB >= 39000 )); then
  DTYPE="bfloat16"
  MAX_LEN=16384
elif (( VRAM_MB >= 23000 )); then
  DTYPE="float16"
  MAX_LEN=8192
else
  DTYPE="float16"
  MAX_LEN=4096
fi
echo "VRAM: ${VRAM_MB} MB → dtype=${DTYPE}, max_len=${MAX_LEN}"

# 6. Launch vLLM
echo ""
echo "--- Starting vLLM server ---"
echo "Access: http://$(curl -s ifconfig.me 2>/dev/null || echo '<IP>'):${PORT}/v1/models"
echo ""

exec $PYTHON -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_LEN}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --gpu-memory-utilization 0.90
```

```bash
# Run it (example: H100 with Llama 3.1 8B)
wget -qO setup.sh https://example.com/gpu_setup.sh  # or scp from your machine
# Or paste directly and:
chmod +x setup.sh
HF_TOKEN="hf_xxxx" ./setup.sh meta-llama/Llama-3.1-8B-Instruct 8000
```

---

### B.14.8 Platform Selection Decision Guide

```
Requirement                              Recommended Platform
─────────────────────────────────────────────────────────────────────
Production, SLA, compliance              AWS EC2 (on-demand)
Cost-sensitive production                AWS EC2 (Spot) or Lambda
Simple setup, reliable H100              Lambda Labs
Cheapest H100 (dev/batch)                Vast.ai (community cloud)
Short experiments, quick teardown        RunPod (per-minute billing)
Serverless batch inference / CI          Modal
Multi-GPU (8× H100)                      AWS p5, Lambda 8×H100
Apple Silicon (M1/M2/M3)                 Local (Metal backend)
No GPU, CPU-only small model             Local llama.cpp or Modal T4
─────────────────────────────────────────────────────────────────────
```

---

## Worked Solutions

### Installation Troubleshooting Q&A

**Q: After `pip install vllm`, `import vllm` fails with "CUDA version mismatch". How do you diagnose and fix this?**

**Step 1 — Check CUDA versions:**
```bash
nvidia-smi              # shows driver CUDA version (upper bound)
nvcc --version          # shows toolkit version (used at compile time)
python -c "import torch; print(torch.version.cuda)"  # PyTorch's CUDA
```

**Step 2 — Identify the mismatch:**
vLLM is compiled against a specific CUDA toolkit version. If PyTorch was installed for CUDA 12.1 but the vLLM wheel was built for CUDA 12.4, the shared libraries won't link correctly.

**Step 3 — Fix:**
```bash
# Uninstall and reinstall with matching CUDA version
pip uninstall vllm torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install vllm  # now downloads wheel matching cu124
```

---

**Q: vLLM starts but crashes with "RuntimeError: CUDA out of memory" immediately on a GPU with 80 GB. The model is LLaMA-3 70B BF16 (140 GB). Why?**

**Root cause:** The model (140 GB) exceeds the GPU's available HBM (80 GB). vLLM tries to load all weights before serving begins.

**Fix options:**

1. Use tensor parallelism: `--tensor-parallel-size 2` on a 2x A100 node (160 GB total).
2. Use quantization: `--quantization fp8` halves model size to 70 GB, fitting on one H100.
3. Use llama.cpp with Q4_K_M (40 GB) if a single GPU is required.

---

**Q: `pip install vllm` on Python 3.13 fails. Why?**

vLLM's supported Python range (as of 2026) is 3.9-3.12. Python 3.13 introduced breaking changes to the C extension API that vLLM's compiled CUDA extensions have not yet been updated to support. Use Python 3.11 (recommended) or 3.12 for production deployments.

---

**Q: After installing ROCm vLLM, `CUDA_VISIBLE_DEVICES=0 python -c "import vllm"` fails but `HIP_VISIBLE_DEVICES=0` succeeds. Why?**

ROCm vLLM uses HIP (Heterogeneous-compute Interface for Portability) internally. `CUDA_VISIBLE_DEVICES` is an NVIDIA CUDA environment variable that ROCm does not honor by default. Use `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` to select AMD GPUs.

---

**Q: The llama.cpp build with `cmake -DGGML_CUDA=ON` fails with "nvcc not found". What is the fix?**

```bash
# Option 1 — Add CUDA toolkit to PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

# Option 2 — Specify nvcc explicitly
cmake -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..

# Option 3 — Use conda's cudatoolkit
conda install -c conda-forge cudatoolkit-dev
```

Verify with: `which nvcc && nvcc --version`

---

**Q: vLLM serving starts successfully but `curl http://localhost:8000/v1/models` returns a connection refused error. What are the three most likely causes?**

1. **vLLM is still loading the model.** Check startup logs — model loading can take 30-120 seconds. The HTTP server only binds after the engine is ready.
2. **Wrong port or host binding.** vLLM defaults to `--host 127.0.0.1`. If running in a Docker container, use `--host 0.0.0.0` to accept external connections.
3. **Firewall or security group blocking port 8000.** Check `netstat -tlnp | grep 8000` to confirm the port is listening, then check firewall rules.

---

**Q: A Docker container running vLLM can see the GPU with `nvidia-smi` but vLLM reports "No GPU found". What is missing?**

The NVIDIA Container Toolkit is not properly configured, or the `--gpus` flag was omitted from `docker run`. Fix:
```bash
# Correct docker run command
docker run --gpus all \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct
```

The `--runtime=nvidia` and `NVIDIA_VISIBLE_DEVICES` environment variable are both required for the CUDA runtime inside the container to find and initialise the GPU.

---

**Q: After upgrading from vLLM 0.5.x to 0.6.x, existing `--engine-args` flags cause startup errors. How do you migrate?**

vLLM 0.6.x reorganised CLI flags, deprecating some and renaming others. Migration steps:
```bash
# Run with --help to see current valid flags
python -m vllm.entrypoints.openai.api_server --help

# Common renames:
# --use-v2-block-manager -> removed (V2 is default in 0.6)
# --worker-use-ray -> --distributed-executor-backend ray
# --swap-space -> still valid but unit changed to GiB

# Check deprecation warnings in startup logs:
vllm serve meta-llama/Llama-3.1-8B 2>&1 | grep -i deprecat
```

Always test flag compatibility in a staging environment before updating production deployments. When in doubt, `vllm serve --help` is the authoritative source — the online changelog at <https://docs.vllm.ai/en/latest/changelog.html> tracks every rename and removal across releases.

*Next: Appendix C covers model loading internals — how vLLM converts HuggingFace weights into its own tensor layout and which dtypes are supported natively versus via automatic conversion.*

