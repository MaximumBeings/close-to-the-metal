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

