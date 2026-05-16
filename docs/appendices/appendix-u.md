# Appendix U — ROCm and AMD GPU Inference

> *"CUDA is not the only way to talk to a GPU. It is just the most popular."*

---

## U.1 Why AMD Matters in 2026

The inference GPU market entered 2026 as a two-player landscape. NVIDIA holds
dominance in training and in the high-end inference segment. AMD's MI300X has
captured a significant and growing share of the hyperscale inference market,
driven by three factors: competitive raw throughput, a substantially lower cost
of ownership, and the tightening export restrictions that have limited H100/H200
availability to certain markets and customers.

For inference engineers, this creates a practical requirement: code and
configurations that work on NVIDIA GPUs should also work — with the right
changes — on AMD GPUs. This appendix provides the translation layer.

---

## U.2 AMD GPU Architecture Overview

### U.2.1 CDNA3: The MI300X Architecture

The MI300X is AMD's data-centre inference GPU, based on the CDNA3 architecture.
Its key specifications:

| Specification | MI300X | H100 SXM5 | Notes |
|---|---|---|---|
| HBM3 capacity | 192 GB | 80 GB | 2.4× more memory |
| Memory bandwidth | 5.3 TB/s | 3.35 TB/s | 1.58× higher |
| FP16 throughput | 1,307 TFLOPS | 989 TFLOPS | 1.32× higher |
| INT8 throughput | 2,614 TOPS | 1,979 TOPS | 1.32× higher |
| FP8 throughput | 5,227 TOPS | 3,958 TOPS | 1.32× higher |
| TDP | 750W | 700W | Slightly higher |
| NVLink equivalent | Infinity Fabric | NVLink 4 | 8× MI300X: 896GB pool |

The 192 GB HBM3 is the headline number for inference. A 70B model at FP16
(~140 GB) fits on a **single MI300X** with 52 GB remaining for KV cache.
The equivalent NVIDIA setup requires 2× H100. This changes the economics and
topology of 70B deployments significantly.

### U.2.2 Memory architecture: Unified vs. discrete

The MI300X uses a **unified CPU-GPU memory architecture**: the CPU and GPU
share the same physical HBM3 memory pool. This means:

1. CPU can directly read/write GPU memory without explicit DMA transfers
2. Model loading is faster (no PCIe copy to GPU memory)
3. Memory spill to CPU RAM is effectively free in terms of bandwidth

```
MI300X unified memory:
  192 GB HBM3 = GPU compute memory = CPU-accessible memory
  No separate CPU DRAM required for model weights
  KV cache, model weights, and CPU tensors share the same pool
```

For very large models (405B), the unified architecture allows spreading weights
across both the MI300X's memory and host memory without the bandwidth penalty
that makes this impractical on NVIDIA.

### U.2.3 RDNA3 and consumer AMD GPUs

RDNA3 (RX 7900 XTX, RX 7800 XT) is AMD's consumer GPU architecture. It lacks
the matrix acceleration units (MFMA instructions) present in CDNA3, making it
significantly less efficient for LLM inference. Performance characteristics:

| GPU | VRAM | FP16 (theoretical) | Practical LLM tok/s (7B, Q4) |
|---|---|---|---|
| RX 7900 XTX | 24 GB | 123 TFLOPS | ~25 tok/s |
| RX 7800 XT | 16 GB | 91 TFLOPS | ~18 tok/s |

Consumer AMD GPUs are supported by llama.cpp (GGML_HIP) but not well-supported
by vLLM. For consumer AMD hardware, llama.cpp is the recommended engine.

---

## U.3 ROCm Platform Overview

ROCm (Radeon Open Compute) is AMD's open-source GPU computing platform,
analogous to CUDA. It provides:

- **HIP** (Heterogeneous-computing Interface for Portability): a CUDA-like C++
  API for writing GPU kernels
- **rocBLAS**: AMD's BLAS library (replaces cuBLAS)
- **hipBLAS**: CUDA-compatible BLAS API wrapping rocBLAS
- **MIOpen**: AMD's deep learning library (replaces cuDNN)
- **rocThrust**: AMD's Thrust replacement
- **hipcc**: CUDA-like compiler driver

The design goal of HIP is source-level compatibility with CUDA: most CUDA code
can be mechanically translated to HIP by replacing `cuda` with `hip` in API
calls. The `hipify-clang` tool automates this.

### U.3.1 CUDA to HIP mapping

```cpp
// CUDA
cudaMalloc(&ptr, size);
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaDeviceSynchronize();

// HIP (drop-in replacement)
hipMalloc(&ptr, size);
hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
hipDeviceSynchronize();
```

The execution model is identical: kernels are written with `<<<grid, block>>>`
syntax (or `hipLaunchKernelGGL`), memory management is explicit, streams
correspond to HIP streams.

```cpp
// CUDA kernel
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// HIP kernel — identical syntax
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < n) c[i] = a[i] + b[i];
}
// Or use CUDA names directly with HIP compatibility headers
```

### U.3.2 ROCm version matrix

| ROCm | CDNA support | Status (2026) |
|---|---|---|
| 5.x | MI100, MI200 | Legacy |
| 6.0 | MI300X (initial) | Stable |
| 6.1 | MI300X, MI300A | Recommended |
| 6.2+ | MI300X, MI325X | Latest |

Always match your ROCm version to your inference engine's requirements. vLLM
0.5+ requires ROCm 6.1+.

---

## U.4 Installing ROCm

### U.4.1 Linux installation (Ubuntu 22.04)

```bash
# Add AMD ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
sudo apt install ./amdgpu-install_6.1.60101-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to render/video groups
sudo usermod -aG render,video $USER

# Verify installation
rocm-smi                    # shows GPU stats
hipcc --version             # HIP compiler version
rocminfo | grep "gfx"       # shows GPU architecture (e.g. gfx942 for MI300X)
```

### U.4.2 Docker (recommended for production)

```dockerfile
# AMD ROCm Docker base image
FROM rocm/dev-ubuntu-22.04:6.1-complete

# Install Python dependencies
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1

# Install vLLM ROCm build
RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.1
```

```bash
# Run container with GPU access
docker run --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --ipc=host --shm-size=16gb \
    -v /models:/models \
    my-rocm-inference-image
```

---

## U.5 vLLM on ROCm

vLLM's ROCm backend is a first-class citizen as of vLLM 0.5. The Python API is
identical to the CUDA backend; only installation and a few configuration
parameters differ.

### U.5.1 Installation

```bash
# Install ROCm-enabled PyTorch first
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.1

# Install vLLM ROCm build
pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.1

# Verify
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should show: AMD Instinct MI300X or similar
```

### U.5.2 Running vLLM on MI300X

```bash
# Llama 3.1 70B on single MI300X (192GB = entire model + 52GB KV cache)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --dtype bfloat16

# Llama 3.1 405B on 3× MI300X (tensor parallel)
# 405B FP16 ≈ 810GB → 3× 192GB = 576GB (use FP8)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 3 \
    --quantization fp8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90

# With ROCm-specific optimisations
VLLM_USE_TRITON_FLASH_ATTN=1 \   # use Triton flash attention for ROCm
HIP_VISIBLE_DEVICES=0,1,2 \      # equivalent of CUDA_VISIBLE_DEVICES
vllm serve meta-llama/Llama-3.1-70B-Instruct ...
```

### U.5.3 ROCm-specific environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HIP_VISIBLE_DEVICES` | all | GPU selection (like CUDA_VISIBLE_DEVICES) |
| `ROCR_VISIBLE_DEVICES` | all | Lower-level GPU selection |
| `HSA_OVERRIDE_GFX_VERSION` | — | Override GPU architecture detection |
| `VLLM_USE_TRITON_FLASH_ATTN` | 0 | Use Triton FA (recommended for ROCm) |
| `PYTORCH_HIP_ALLOC_CONF` | — | Memory allocator config |
| `HIP_FORCE_DEV_KERNARG` | 1 | Workaround for some kernel argument bugs |

### U.5.4 Performance gap: ROCm vs CUDA

As of ROCm 6.1 / vLLM 0.5, throughput comparison for Llama 3.1 70B:

| Metric | H100 SXM5 × 2 | MI300X × 1 | Notes |
|---|---|---|---|
| Decode throughput (batch=32) | 2,100 tok/s | 1,850 tok/s | -12% |
| TTFT (2K input, batch=1) | 95ms | 110ms | -16% |
| Memory bandwidth utilisation | 82% | 79% | Similar |
| Model fits on single device | ✗ (140GB > 80GB) | ✓ (140GB < 192GB) | MI300X advantage |

The throughput gap has narrowed significantly through 2025 as AMD invested in
ROCm kernel optimisation. For the 70B use case specifically, the MI300X single-
card solution is often preferable to dual-H100 due to reduced communication
overhead.

---

## U.6 llama.cpp on ROCm/HIP

llama.cpp supports AMD GPUs via the `GGML_HIP` backend. This works on both
data-centre (MI300X) and consumer (RX 7900 XTX) AMD GPUs.

### U.6.1 Building llama.cpp with HIP

```bash
# Prerequisite: ROCm 6.1+ installed
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with HIP support
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx942 \   # MI300X target; use gfx1100 for RX 7900 XTX
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Check target architecture for your GPU
rocminfo | grep "gfx"
# MI300X: gfx942
# RX 7900 XTX: gfx1100
# MI250X: gfx90a
# Multiple GPUs: -DAMDGPU_TARGETS="gfx942;gfx90a"
```

### U.6.2 Running on AMD GPU

```bash
# MI300X: offload all layers
./build/bin/llama-server \
    --model Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 80 \    # all 80 layers to MI300X
    --ctx-size 32768 \
    --port 8080

# RX 7900 XTX (24GB): model at Q4_K_M fits
./build/bin/llama-server \
    --model Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 33 \
    --ctx-size 8192 \
    --port 8080

# Verify GPU is being used
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    --model model.gguf \
    --n-gpu-layers 99 \
    --prompt "Hello, world!" \
    -ngl 99 -v 2>&1 | grep "ggml_hip"
```

### U.6.3 Multi-GPU with llama.cpp on ROCm

```bash
# Distribute 405B across 4× MI300X
HIP_VISIBLE_DEVICES=0,1,2,3 ./build/bin/llama-server \
    --model Meta-Llama-3.1-405B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 126 \
    --split-mode row \
    --ctx-size 16384 \
    --port 8080
```

---

## U.7 Profiling on ROCm

### U.7.1 rocm-smi — GPU status

```bash
rocm-smi                    # shows all GPUs
rocm-smi --showmeminfo vram # VRAM usage
rocm-smi --showuse          # GPU utilisation
rocm-smi --showtemp         # temperature
rocm-smi --showpower        # power consumption

# Continuous monitoring
watch -n 1 rocm-smi
```

### U.7.2 rocprof — kernel profiling

`rocprof` is the ROCm equivalent of Nsight/ncu. It profiles HIP kernels and
gives the same metrics (FLOPS, memory bandwidth, occupancy) as CUDA profilers.

```bash
# Profile a vLLM inference run
rocprof --stats \               # summary statistics
        --hip-trace \           # trace HIP API calls
        -o profile_output.csv \
        python inference_script.py

# View top kernels by time
rocprof --stats python ... 2>&1 | grep -A 20 "Top Kernels"
```

```python
# Python profiling with AMD's rocm_smi_lib
import amdsmi

amdsmi.amdsmi_init()
handles = amdsmi.amdsmi_get_processor_handles()
h = handles[0]

# Poll GPU metrics during inference
metrics = amdsmi.amdsmi_get_gpu_metrics_info(h)
print(f"GPU utilization: {metrics['average_gfx_activity']}%")
print(f"Memory used:     {metrics['current_vram_used']} MB")
print(f"Temperature:     {metrics['current_socket_power']}W")
```

### U.7.3 hipprof — HIP-level timeline

For detailed kernel timelines analogous to Nsight Systems:

```bash
hipprof --tool trace \
        --hip-api \
        --output timeline.json \
        python inference_script.py

# View in perfetto or chrome://tracing
```

---

## U.8 Porting CUDA Kernels to HIP

If you are writing custom inference kernels for AMD, the `hipify-clang` tool
automates most of the CUDA → HIP translation:

```bash
# Install hipify
sudo apt install hipify-clang

# Convert a CUDA file to HIP
hipify-clang cuda_kernel.cu --cuda-path=/usr/local/cuda \
             -o hip_kernel.hip

# Common patterns after hipify
# cuda* → hip*
# cub:: → hipcub::
# __ldg() → __ldg() (same)
# cooperative_groups → hip cooperative groups
```

### U.8.1 Matrix multiply: WMMA → MFMA

NVIDIA's WMMA (Warp Matrix Multiply-Accumulate) API has an AMD equivalent in
MFMA (Matrix Fused Multiply-Add). The concepts are analogous but the tile
sizes and API differ:

```cpp
// NVIDIA WMMA (A100: 16×16×16 tiles)
#include <mma.h>
using namespace nvcuda::wmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;
fill_fragment(c_frag, 0.0f);
load_matrix_sync(a_frag, a_ptr, 16);
load_matrix_sync(b_frag, b_ptr, 16);
mma_sync(c_frag, a_frag, b_frag, c_frag);

// AMD MFMA (MI300X: 32×32×8 or 16×16×4 tiles)
// Lower-level: use builtins directly
float16_t a[4], b[4];
float c[16] = {0};
// v_mfma_f32_16x16x16f16: 16×16×16 tile
c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
```

For production use, avoid raw MFMA builtins. Use:
- **rocBLAS** for standard GEMM (equivalent to cuBLAS)
- **hipBLASLt** for LT (Cublaslt) equivalent — tensor-op GEMM with epilogue fusion
- **CK (Composable Kernel)** — AMD's CUTLASS equivalent for custom fused kernels

---

## U.9 CK (Composable Kernel) — AMD's CUTLASS

Composable Kernel is AMD's library for high-performance, composable GPU kernels
on ROCm. It is the AMD equivalent of CUTLASS (Appendix Q).

```cpp
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using F16 = ck::half_t;
using F32 = float;
using RowMajor = ck::tensor_layout::gemm::RowMajor;
using ColMajor = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// Create a GEMM: C = A × B (FP16 → FP32)
auto gemm = DeviceGemmInstance{};
auto invoker = gemm.MakeInvoker();
auto argument = gemm.MakeArgument(
    a_device_buf.GetDeviceBuffer(),
    b_device_buf.GetDeviceBuffer(),
    c_device_buf.GetDeviceBuffer(),
    M, N, K,
    StrideA, StrideB, StrideC,
    PassThrough{}, PassThrough{}, PassThrough{}
);
invoker.Run(argument, StreamConfig{nullptr, false});
```

CK is what vLLM ROCm uses internally for its fused attention and GEMM kernels.

---

## U.10 Common Issues and Fixes

### Issue: Kernel compilation slow on first run

ROCm compiles GPU kernels on-demand for the detected GPU architecture. This
means the first inference run is slow (30–120 seconds for kernel JIT
compilation). Fix: pre-cache kernels.

```bash
# Pre-warm the kernel cache
python -c "
import torch
import vllm
llm = vllm.LLM('meta-llama/Llama-3.1-8B', gpu_memory_utilization=0.5)
llm.generate(['warmup'], vllm.SamplingParams(max_tokens=1))
print('Kernel cache warmed')
"
# Subsequent runs skip JIT compilation
```

### Issue: `HSA_STATUS_ERROR_OUT_OF_RESOURCES`

This typically means GPU memory fragmentation. Fix: set explicit memory limits:

```bash
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
# Or reduce gpu_memory_utilization in vLLM
```

### Issue: Triton kernels not found for ROCm

vLLM uses Triton for some kernels (flash attention, activation functions). The
ROCm version of Triton is a separate package:

```bash
pip install triton-rocm
# Verify
python -c "import triton; print(triton.__version__)"
```

### Issue: Wrong AMDGPU_TARGETS for llama.cpp

If you build for `gfx942` but your GPU is `gfx90a` (MI250X), the binary will
run but performance will be poor (wrong instruction scheduling). Always check:

```bash
rocminfo | grep "gfx"
# Then rebuild with the correct target
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a ...
```

---

## U.11 MI300X vs H100: Decision Guide

| Scenario | Recommend | Reason |
|---|---|---|
| Serve 70B model, budget-constrained | MI300X × 1 | Single card vs dual-H100 |
| Maximum throughput, unlimited budget | H100 × 2+ | Kernel maturity, NVLink |
| 405B in 3 cards | MI300X × 3 | 576GB pool; H100 needs 6+ cards |
| Custom CUDA kernels (existing code) | H100 | CUDA ecosystem matures faster |
| New deployment, AMD pricing 30% lower | MI300X | ROCm parity at 6.1+ |
| Inference-only, no training | Either | Performance parity within ~15% |
| Training + inference | H100 | CUDA training ecosystem superior |

---

## U.12 Self-Check Questions

1. The MI300X has 192 GB HBM3. A Llama 3.1 70B model in BF16 occupies ~140 GB.
   Using `gpu_memory_utilization=0.85`, how many GB are available for KV cache?
   At 320 KB/token (70B BF16 KV cache), how many maximum concurrent 4K-token
   requests can fit?

2. Explain the difference between `HIP_VISIBLE_DEVICES` and
   `ROCR_VISIBLE_DEVICES`. In what scenario would they give different results?

3. `hipify-clang` translates `cudaMalloc` to `hipMalloc`. Describe one CUDA
   feature that has no direct HIP equivalent and requires manual porting.

4. The MFMA instruction on MI300X uses 32×32×8 tiles for FP16. NVIDIA's WMMA
   uses 16×16×16 tiles. How does this tile size difference affect the minimum
   batch size needed to fully utilise the matrix units? Which is more efficient
   for batch size 1 (single-token decode)?

5. A company runs their LLM inference on 4× H100 nodes and wants to migrate to
   MI300X. Their current setup uses custom CUDA C++ kernels for fused attention.
   List the steps required to port those kernels to ROCm, in order. Which step
   is most likely to require manual intervention rather than automated tooling?
