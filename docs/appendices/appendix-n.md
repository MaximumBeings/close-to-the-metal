# Appendix N — Edge Inference on Linux SBCs: Raspberry Pi and NVIDIA Jetson

> *"These are not toy deployments. A Raspberry Pi 5 can serve a quantized 7B model to a local home-automation stack. A Jetson Orin AGX can serve a 70B model with GPU acceleration in a 60-watt envelope. The physics is real; only the scale changes."*

---

## N.1 What Is an SBC and Why Does It Matter for LLM Inference?

An **SBC (Single-Board Computer)** is an entire computer — CPU, RAM, storage controller, network interface, and sometimes a GPU — built onto a single circuit board roughly the size of a credit card. The Raspberry Pi is the most recognisable example. NVIDIA's Jetson line is the professional-grade alternative, adding a programmable GPU to the same compact form factor.

SBCs matter for LLM inference for three reasons. First, **cost**: a Raspberry Pi 5 costs around $80, compared to thousands of dollars for a cloud GPU instance running around the clock. For applications that need always-on private inference — home automation, local assistants, document processing — the economics strongly favour an on-premises SBC over rented cloud compute. Second, **privacy**: inference running on hardware you physically own means your prompts and responses never leave your network. Third, **edge deployment**: a Jetson Orin installed in a vehicle, a factory floor, or a medical device can run ML inference entirely offline, with no dependency on network connectivity or cloud availability.

This appendix covers two families. The **Raspberry Pi** (versions 4 and 5) is the general-purpose SBC — inexpensive, widely available, runs standard Debian Linux, and performs CPU-only inference with an optional Vulkan GPU backend on the Pi 5. The **NVIDIA Jetson** family (Nano through AGX Orin) is the professional edge-AI platform — it runs a CUDA-capable GPU on the same module as the CPU, enabling full GPU-accelerated inference in a compact, low-power form factor.

### N.1.1 The Unified Memory Advantage — Shared by All SBCs

One property that all SBCs in this appendix share — and that they share with Apple Silicon (Appendix M) — is **unified memory**: the CPU and GPU (where present) access the same physical RAM pool. There is no separate VRAM, no PCIe bus to transfer weights from system RAM to GPU memory.

On a desktop PC with a discrete GPU, loading a model involves two steps: the CPU loads the GGUF file from disk into system RAM, then copies the GPU-assigned layers over the PCIe bus into the GPU's dedicated VRAM. PCIe Gen 4 x16 bandwidth is about 32 GB/s — fast but finite, and it introduces a one-time transfer cost at load time plus ongoing KV cache traffic during inference.

On an SBC, there is only one memory pool. The CPU loads the file from storage directly into the shared RAM, and the GPU reads that RAM at the same bandwidth as the CPU — no copying, no bus transfer. This matters practically because it means the GPU can access 100% of the system RAM for model weights, rather than being constrained to a physically separate VRAM pool.

---

## N.2 Part One: Raspberry Pi

### N.2.1 Choosing the Right Pi Model

Not all Raspberry Pi models are equally capable of running LLMs. The performance gap between generations is large enough that the choice of hardware significantly affects whether the result is usable or merely technically functional.

**The Raspberry Pi 4** uses the Cortex-A72 CPU core, a 2012 Arm architecture design that lacks the matrix-multiply acceleration extensions added in later generations. Running a Q4_K_M 7B model on a Pi 4 produces 1–2 tokens per second — slow enough to be frustrating in an interactive application. The Pi 4 can serve as a model validation device (confirming a model loads and produces sensible output) but is not suitable for anything requiring a responsive experience.

**The Raspberry Pi 5** uses the Cortex-A76 core, a 2018 design with significantly wider SIMD execution units and better branch prediction. For the matrix-multiply operations that dominate LLM decode, the A76 is roughly 2–3× faster than the A72 — lifting the same Q4_K_M 7B model from 1–2 tokens/sec to 4–8 tokens/sec. That is the difference between an unusable demo and a genuinely useful local assistant.

The **Pi Zero 2W** has only 512 MB of RAM — far too little to load any model of practical size. Even the smallest useful model (a 1B parameter model at Q4_K_M) requires about 800 MB. The Pi Zero 2W cannot be used for LLM inference.

**Minimum viable setup for serious work:** Pi 5 8 GB + active cooling (a fan heatsink, not just a passive heatsink) + NVMe SSD via the Pi 5's PCIe M.2 HAT (for fast model loading) + the official 27W USB-C power supply. Each component matters: 8 GB leaves room for a useful context window, active cooling prevents thermal throttling, NVMe cuts model load time from 60+ seconds to under 15 seconds, and the 27W supply ensures the Pi does not brown out under CPU load.

### N.2.2 Operating System — Why 64-Bit Matters

The Raspberry Pi OS comes in 32-bit and 64-bit variants. For LLM inference, you must use the **64-bit version**. The reasons are both practical and architectural.

**RAM addressing:** The 32-bit OS caps addressable memory at approximately 3 GB, even on an 8 GB Pi. A Q4_K_M 7B model requires about 4 GB at runtime (weights plus KV cache). Running a 32-bit OS means you cannot fit the model in memory at all.

**SIMD register width:** Arm's NEON SIMD instructions — the vectorised operations that accelerate matrix multiply — operate on 128-bit registers. In 32-bit mode, each NEON register can hold two 64-bit values. In 64-bit mode (AArch64), the same NEON instructions use 128-bit registers capable of holding four 32-bit values simultaneously. llama.cpp's inner loops are tuned for AArch64 NEON — running in 32-bit mode leaves half the SIMD hardware idle.

```bash
# Verify you are on a 64-bit kernel before proceeding
uname -m
# Correct output: aarch64
# Wrong output:   armv7l  ← this means 32-bit; re-flash with 64-bit OS

# Update everything before building — stale packages cause subtle build failures
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y cmake git build-essential python3-dev \
    libopenblas-dev pkg-config
```

### N.2.3 Swap Space — Why It Matters and How to Set It Up

**What swap is:** Swap is disk space that the OS uses as a slow extension of RAM. When physical RAM fills up, the OS moves some pages from RAM to swap (the swap-out operation) and later reads them back (the swap-in operation). Swap space is 10–100× slower than RAM, so heavy swap use causes severe performance degradation — but it prevents the OOM (out of memory) killer from terminating processes.

For LLM inference on a Pi, the concern is not that you want to run models that do not fit in RAM — you should always choose a model that fits. The concern is that **inference is not the only thing using RAM**. The OS itself uses roughly 500 MB. Background services, the SSH session, Python utilities — these accumulate. A momentary spike during model loading can push total usage above the physical RAM limit and cause the inference process to be killed mid-operation. Adding swap protects against this.

**Key rule:** Put swap on the fastest storage you have, not the SD card. SD cards have write endurance limits, and swap workloads involve many small random writes that degrade SD cards quickly. Put swap on the NVMe drive.

```bash
# Create a 4 GB swap file on the NVMe drive (mounted at /mnt/nvme here)
sudo fallocate -l 4G /mnt/nvme/swapfile
sudo chmod 600 /mnt/nvme/swapfile   # swap files must be readable only by root
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile

# Make it persist across reboots by adding to /etc/fstab
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Set swappiness to 10 — this tells the kernel to prefer keeping pages
# in RAM and only use swap under real pressure. The default of 60 causes
# the kernel to swap too eagerly for an inference workload.
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### N.2.4 Building llama.cpp on Raspberry Pi

**What OpenBLAS is:** BLAS (Basic Linear Algebra Subprograms) is a specification for a library of fundamental matrix operations — addition, multiplication, dot products. OpenBLAS is a high-performance open-source implementation of this specification, optimised for each CPU architecture it runs on. llama.cpp uses OpenBLAS for the batch matrix-multiply operations in the prefill phase (processing the input prompt). Without it, prefill is noticeably slower.

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CPU-only build with OpenBLAS acceleration
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BLAS=ON \                # use OpenBLAS for batched matrix ops
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_NATIVE=ON               # compile for the exact CPU running the build
                                   # enables whatever SIMD extensions this Pi has

cmake --build build --config Release -j4
# -j4 uses all 4 CPU cores for parallel compilation
# Build time: 8–12 minutes on Pi 5
```

`-DGGML_NATIVE=ON` deserves explanation: normally, a compiler generates code that runs on the minimum supported CPU. `NATIVE` tells it to generate code optimised for the exact CPU it is running on right now — enabling NEON intrinsics, SVE if available, and any other extensions the hardware supports. **Never use `NATIVE` for cross-compiled builds** (building on one machine for another), because the resulting binary will crash on a CPU that does not support all the same extensions.

```bash
# Verify the build succeeded and check which backends are active
./build/bin/llama-cli --version
# Expected: version string mentioning CPU, BLAS
# Red flag: if it mentions "no BLAS" after you enabled it, something went wrong
```

### N.2.5 Vulkan Backend on Pi 5

**What Vulkan is and how it differs from CUDA:** CUDA is NVIDIA's proprietary GPU programming interface — it only works on NVIDIA GPUs. Vulkan is an open, cross-platform GPU API that works on a much broader range of GPUs, including AMD, Intel, ARM, and the VideoCore VII GPU in the Raspberry Pi 5. Vulkan is lower-level than CUDA — it is closer to directly programming the GPU hardware — and achieving peak performance requires more explicit management of GPU memory and synchronisation.

llama.cpp has a Vulkan backend that offloads matrix operations to any Vulkan-capable GPU. On the Pi 5's VideoCore VII, this does not deliver the dramatic speedups you see on a dedicated gaming GPU — the VideoCore VII is a modest GPU designed primarily for video decoding and display, not compute workloads. However, offloading a moderate number of layers (8–16 out of 32 for a 7B model) frees the CPU from those matrix multiplications, which can reduce thermal load and improve sustained throughput.

Think of it as a workload distribution strategy rather than a raw acceleration strategy: the CPU and GPU are doing different work in parallel, keeping both utilised rather than bottlenecking everything on the CPU alone.

```bash
# Install Vulkan development libraries
sudo apt install -y libvulkan-dev vulkan-tools glslc

# Confirm VideoCore VII is visible as a Vulkan device
vulkaninfo --summary 2>/dev/null | grep "GPU id"
# Expected: Broadcom Limited V3D 7.x.x

# Build with Vulkan backend enabled
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \   # enable Vulkan compute backend
    -DGGML_NATIVE=ON
cmake --build build --config Release -j4
```

```bash
# Run with GPU layer offloading
./build/bin/llama-server \
    --model /mnt/nvme/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 16 \   # send 16 of 28 layers to VideoCore VII GPU
    --ctx-size 2048 \
    --host 0.0.0.0 \
    --port 8080

# Watch startup output for GPU allocation confirmation:
# "GGML_VULKAN: Allocated ..." means the GPU backend is active
```

**Calibrating `--n-gpu-layers` on Pi 5:** Start at 0 and increase by 4. Each time you increase it, watch the startup log for Vulkan allocation messages and run a brief benchmark. Stop increasing when the performance gain flattens or when you start seeing memory pressure warnings. For a Q4_K_M 7B model on Pi 5 8 GB, `-ngl 20` is a reasonable starting point: it offloads the computationally intensive upper layers while leaving enough RAM for the OS and KV cache.

### N.2.6 Model Storage — Why SD Cards Are Not Enough

Model files are large and must be read sequentially at startup. The time from `llama-server` launch to first token is dominated by model loading speed for large models. Here is what that looks like in practice:

| Storage type | Sequential read speed | Time to load 4.4 GB model |
|---|---|---|
| SD Card (Class 10) | 20–40 MB/s | 90–200 seconds |
| USB 3.0 SSD (generic) | 200–400 MB/s | 11–22 seconds |
| NVMe via Pi 5 PCIe M.2 HAT | 400–900 MB/s | 5–11 seconds |

For an always-on service, startup time matters mainly after reboots. For interactive development — loading different models, trying different quantizations — the 2-minute SD card load time becomes a significant friction cost.

```bash
# Benchmark your storage to know what you're working with
dd if=/mnt/nvme/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
   of=/dev/null bs=1M status=progress
```

The Pi 5 has a single PCIe Gen 2 lane exposed through its M.2 HAT connector. You can unlock Gen 3 speed — nearly doubling bandwidth — by adding one line to the boot config:

```bash
# Add to /boot/firmware/config.txt
# Then reboot
echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

Verify Gen 3 is active after reboot:
```bash
lspci -vv | grep -i "lnksta"
# Should show: Speed 8GT/s (Gen 3), not 5GT/s (Gen 2)
```

### N.2.7 Quantization Tiers for Raspberry Pi

Choosing a quantization level is a three-way trade-off between model quality, file size, and inference speed. On a Pi, the dominant constraint is RAM — you need the model weights plus KV cache plus OS overhead to fit within available memory.

| Available RAM | Model | Recommended Quant | File Size | Decode speed (Pi 5) | Notes |
|---|---|---|---|---|---|
| 8 GB | Llama-3.2-3B | Q8_0 | 3.3 GB | 7–9 tok/s | Near-full quality |
| 8 GB | Llama-3.2-3B | Q4_K_M | 1.9 GB | 9–12 tok/s | Best quality/speed |
| 8 GB | Qwen2.5-7B | Q4_K_M | 4.4 GB | 4–6 tok/s | Tight at 8K context |
| 8 GB | Qwen2.5-7B | Q2_K | 2.7 GB | 5–8 tok/s | Noticeable quality loss |
| 8 GB | Llama-3.1-8B | Q4_K_M | 4.9 GB | 3–5 tok/s | Barely fits; 2K ctx only |
| 4 GB | Llama-3.2-1B | Q4_K_M | 0.8 GB | 12–18 tok/s | For Pi with 4 GB |
| 4 GB | Phi-3-mini (3.8B) | Q4_K_M | 2.2 GB | 6–9 tok/s | Strong small model |

**A practical memory budget example:** Pi 5 8 GB, running Qwen2.5-7B-Q4_K_M, 4096 token context:

- OS and background services: ~500 MB
- Model weights (Q4_K_M 7B): ~4.4 GB
- KV cache (2 × 28 layers × 32 heads × 128 dim × 4096 ctx × 2 bytes): ~1.5 GB
- Inference buffers: ~0.5 GB
- **Total: ~7.0 GB** — fits with ~1 GB to spare

Increasing context to 8192 tokens doubles the KV cache to ~3.0 GB, pushing total to ~8.5 GB — which exceeds the Pi 5's RAM. So for a 7B model on Pi 5, 4096 tokens is the practical context ceiling.

### N.2.8 Thermal Management — The Pi 5 Throttling Problem

Thermal throttling is what happens when a CPU detects its temperature is approaching a dangerous level and reduces its clock speed to generate less heat. For an LLM inference workload — which is one of the most sustained compute-intensive workloads a Pi will ever see — throttling is not a theoretical concern. It happens, and it matters.

**The throttle sequence on Pi 5:** At normal temperatures (under 60°C with good cooling), the Cortex-A76 runs at its full 2.4 GHz boost clock. As temperature rises to 80°C, the kernel reduces the clock to 1.5 GHz. At 85°C it throttles further. At 90°C the Pi hard-shuts down to protect the hardware. A passive heatsink (no fan) on a Pi 5 running LLM inference will reach the throttle threshold within 5–10 minutes — after which the effective tokens/sec drops by 30–50%.

This means a 30-second benchmark may look significantly better than 10-minute sustained inference. Always benchmark over at least 3 minutes and report the average over the last half of the run, when the temperature has stabilised.

```bash
# Monitor temperature and throttle status in real time
watch -n 2 "vcgencmd measure_temp && vcgencmd get_throttled"

# The get_throttled output is a bitmask:
# 0x0      — no throttling, no history
# 0x50000  — previously throttled (currently OK, but was throttled)
# 0x50005  — currently throttled AND frequency capped (active thermal issue)
# Any nonzero value during inference = your cooling is insufficient
```

**Cooling recommendations:**

- *Passive heatsink only:* Reaches throttle in 5–10 minutes. Acceptable only for occasional short inference sessions.
- *Official Pi 5 Active Cooler (fan + heatsink):* Maintains temperature below 70°C during sustained inference. Minimum viable for a production deployment.
- *Enclosed case with fan (Argon NEO 5, Pimoroni NVMe Base):* Maintains 60–65°C. Recommended for an always-on inference server.

Separately, setting the CPU governor to `performance` mode prevents the kernel from reducing clock speed proactively when load is low, ensuring every token is generated at full speed:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Verify the change took effect
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
# Should show a high number (e.g. 2400000 = 2.4 GHz)
```

### N.2.9 Running llama-server as a systemd Service

**What systemd is:** systemd is the service manager for most modern Linux distributions, including Raspberry Pi OS. It replaces the older `init` system and provides a consistent way to define background services, specify startup dependencies, configure automatic restart on failure, and manage logging. You define a service by writing a `.service` file and placing it in `/etc/systemd/system/`.

When you define `llama-server` as a systemd service, it starts automatically at boot, restarts if it crashes, and logs all output to the system journal (accessible with `journalctl`). This is the difference between a service you have to manually start after every reboot and one that is always there when you need it.

```ini
# /etc/systemd/system/llama-server.service
[Unit]
Description=llama.cpp inference server
# 'After=network.target' means systemd will not start this service
# until the network is up — important if the server binds to an
# IP address that may not exist at early boot.
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi     # run as the pi user, not root, for security

# The full command to start the server. Each argument is on its own line
# for readability — systemd accepts this format.
ExecStart=/home/pi/llama.cpp/build/bin/llama-server \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --ctx-size 4096 \
    --n-gpu-layers 20 \
    --parallel 1 \
    --host 0.0.0.0 \
    --port 8080 \
    --log-disable

# Restart the service if it exits for any reason (crash, OOM kill, etc.)
Restart=on-failure
RestartSec=5     # wait 5 seconds before restarting to avoid tight loops

StandardOutput=journal
StandardError=journal
LimitNOFILE=65536   # allow many open file descriptors for HTTP connections

[Install]
# WantedBy=multi-user.target means: start this service when the system
# reaches the normal multi-user runlevel (i.e., normal boot)
WantedBy=multi-user.target
```

```bash
# Load the service definition and enable auto-start at boot
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Check current status
sudo systemctl status llama-server

# View live logs
journalctl -u llama-server -f

# Verify the server is responding
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

### N.2.10 Realistic Performance Expectations

A Pi 5 running a Q4_K_M 7B model will deliver approximately **4–6 tokens per second** in sustained decode. This is enough for:

- A single-user local chatbot
- Home-automation natural-language command parsing (where a 1–2 second response is acceptable)
- Batch document summarisation running overnight
- Private, offline processing of sensitive text

It is **not enough** for multi-user interactive serving, real-time transcription, or any application requiring sub-second time-to-first-token on long prompts. For those requirements, move to the Jetson platform (N.3) or a data-center GPU.

A practical pattern for teams that need more throughput on a Pi budget: run 4× Pi 5 units behind an nginx load balancer. Each Pi handles one concurrent request; the aggregate throughput is 16–24 tokens/sec and the hardware cost is around $400, roughly the same as a single Jetson Orin Nano.

---

## N.3 Part Two: NVIDIA Jetson

### N.3.1 Understanding the Jetson Product Line

NVIDIA's Jetson family is not one product — it is a spectrum of compute modules spanning from a $200 entry device to a $499 developer kit capable of running 70B models. The key differentiator between models is the GPU: Jetson's GPU is fully CUDA-capable, meaning the same CUDA kernels that run on data-center A100s also run on a Jetson Orin — just more slowly, proportional to the number of CUDA cores and memory bandwidth available.

| Module | GPU Arch | CUDA Cores | Memory | BW | TDP | Price |
|---|---|---|---|---|---|---|
| Jetson Nano (2019, discontinued) | Maxwell | 128 | 4 GB LPDDR4 | 25.6 GB/s | 5–10 W | — |
| Jetson Orin Nano 4GB | Ampere | 512 | 4 GB LPDDR5 | 34 GB/s | 7–10 W | $149 |
| Jetson Orin Nano 8GB | Ampere | 1024 | 8 GB LPDDR5 | 68 GB/s | 7–15 W | $199 |
| Jetson Orin NX 8GB | Ampere | 1024 | 8 GB LPDDR5 | 102 GB/s | 10–20 W | $299 |
| Jetson Orin NX 16GB | Ampere | 1024 | 16 GB LPDDR5 | 102 GB/s | 10–25 W | $399 |
| Jetson AGX Orin 32GB | Ampere | 2048 | 32 GB LPDDR5 | 204 GB/s | 15–60 W | $399 kit |
| Jetson AGX Orin 64GB | Ampere | 2048 | 64 GB LPDDR5 | 204 GB/s | 15–60 W | $499 kit |

**A note on the original Jetson Nano:** The 2019 Jetson Nano uses Maxwell GPU architecture and CUDA 10.2. Maxwell is two GPU generations older than the Ampere used in the Orin series. Many modern llama.cpp CUDA optimisations — including Flash Attention — require CUDA capabilities not present in Maxwell. The original Nano can still run llama.cpp in CPU mode or with limited CUDA offloading, but its performance is comparable to a Raspberry Pi 5 at roughly 2.5× the cost. For new projects, do not buy the original Nano.

**The recommended entry point is the Jetson Orin Nano 8 GB.** It has 1024 Ampere CUDA cores, 8 GB of LPDDR5 at 68 GB/s bandwidth, and supports the full range of llama.cpp CUDA optimisations including Flash Attention. It delivers 15–22 tokens/sec for a Q4_K_M 7B model — 3–5× the throughput of a Pi 5, at 2.5× the cost.

### N.3.2 JetPack — The Software Foundation

**What JetPack is and why you must use it:** JetPack is NVIDIA's board support package for Jetson modules. It bundles an Ubuntu-based OS with NVIDIA's proprietary kernel patches, CUDA toolkit, cuDNN (deep learning primitives), TensorRT, and the hardware drivers for Jetson's camera, display, and peripheral interfaces. Critically, the CUDA driver and the GPU are physically inseparable — without NVIDIA's patched kernel, the GPU simply cannot be accessed. You cannot install a generic Ubuntu and expect CUDA to work.

**This is the most common mistake new Jetson users make:** flashing a standard Ubuntu image and then wondering why CUDA is unavailable. Always start with JetPack.

JetPack is flashed from a host x86-64 Ubuntu machine using NVIDIA's SDK Manager application, connected to the Jetson module via USB:

```bash
# On host machine (must be x86-64 Ubuntu) — install SDK Manager
# Download from: https://developer.nvidia.com/sdk-manager
# Then flash JetPack 6.0 to Jetson Orin Nano:
sdkmanager --cli install \
    --product Jetson \
    --target JETSON_ORIN_NANO_8GB \
    --version 6.0 \
    --select-default
```

**Check your JetPack version on a running Jetson:**

```bash
cat /etc/nv_tegra_release
# Output example: # R35 (release), REVISION: 3.1 → JetPack 5.1.1
# R36 → JetPack 6.0

dpkg -l | grep jetpack
# Shows the installed JetPack meta-package version
```

For llama.cpp with CUDA, you need JetPack 5.x (for Orin with CUDA 11.4) or JetPack 6.x (CUDA 12.x). The original Nano supports JetPack 4.6 (CUDA 10.2) only.

### N.3.3 Building llama.cpp with CUDA on Jetson

The CUDA build process on Jetson is nearly identical to a desktop Linux CUDA build, with one critical difference: the `CMAKE_CUDA_ARCHITECTURES` flag must match the GPU in the specific Jetson module.

**What the CUDA architecture flag means:** NVIDIA GPUs have different instruction sets across generations, identified by a "compute capability" version (like `sm_87` for Jetson Orin's Ampere GPU). A CUDA binary compiled for `sm_80` (A100) will not run correctly on `sm_87` (Jetson Orin) because the instruction sets differ. If you use the wrong flag, you will get one of two failure modes: a cryptic `CUDA error: no kernel image is available for execution on the device` error, or silent correctness errors where the GPU falls back to an older code path and produces wrong results.

| Jetson Module | GPU Architecture | `CMAKE_CUDA_ARCHITECTURES` |
|---|---|---|
| Jetson Nano (original) | Maxwell | `53` |
| Jetson Xavier NX | Volta | `72` |
| Jetson Orin Nano / NX / AGX | Ampere | `87` |

```bash
# Install build dependencies
sudo apt update
sudo apt install -y cmake git build-essential python3-pip \
    libopenblas-dev pkg-config ninja-build

# Confirm CUDA is present — this should print a version string
nvcc --version
# If this fails, JetPack is not properly installed

# Monitor GPU during the build (open in a second terminal)
# tegrastats is Jetson's equivalent of nvidia-smi
# 'tegrastats' shows GPU utilisation, temperature, power draw
watch -n 2 tegrastats

# Clone and build
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \   # MUST match your Jetson's GPU
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_NATIVE=ON \
    -GNinja

cmake --build build --config Release -j$(nproc)
# Build time: 15–25 minutes (CUDA kernel compilation is the bottleneck)
```

**Verify CUDA is active after building:**

```bash
./build/bin/llama-cli --version
# Should include "CUDA" in the backend list
# Should NOT say "CPU only"

# Smoke test: load a model and check for CUDA allocation messages
./build/bin/llama-cli \
    --model /mnt/nvme/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 99 \
    -p "Once upon a time" \
    -n 10
# Watch startup output for: "CUDA0: ..." allocation messages
# If you see "CPU only" during load, CUDA is not working
```

**For the original Jetson Nano (Maxwell, CUDA 10.2, sm_53):**

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=53 \   # Maxwell
    -DGGML_NATIVE=ON \
    -GNinja
```

Note that Flash Attention (`--flash-attn`) is not supported on Maxwell. The flag will be silently ignored or produce a warning.

### N.3.4 Power Mode Configuration — Unlocking Full Performance

**What power modes are:** Jetson modules have configurable power envelopes. In the lowest-power mode, the CPU and GPU run at reduced clock speeds to stay within a watt budget — useful for battery-powered deployments but harmful for inference throughput. In the maximum performance mode, all cores and the GPU run at their peak frequencies.

`nvpmodel` is the tool that switches between these modes. `jetson_clocks` is a companion tool that explicitly locks clocks at their maximum frequency, preventing the power governor from reducing them even within the maximum power mode.

```bash
# List available power modes for your specific board
sudo nvpmodel -q --verbose
# Output shows mode IDs and their CPU/GPU frequency settings

# Common mode IDs (verify with the command above — they vary by board):
# Jetson Orin Nano: 0 = MAXN (15W), 1 = 10W, 2 = 7W
# Set maximum performance mode
sudo nvpmodel -m 0

# Lock CPU and GPU clocks at maximum frequency
# Without this, the power governor can reduce clocks even in mode 0
sudo jetson_clocks

# Verify clocks are locked
sudo jetson_clocks --show
# Look for CPU and GPU frequencies at their maximum values
```

**Important:** `jetson_clocks` does not persist across reboots. Without a startup service to re-run it, your Jetson will boot in a lower-performance mode. Make it permanent with a systemd service:

```bash
# Create a systemd service to lock clocks at maximum on every boot
sudo tee /etc/systemd/system/jetson-clocks.service > /dev/null << 'EOF'
[Unit]
Description=Lock Jetson CPU and GPU clocks at maximum frequency
After=nvpmodel.service   # run after nvpmodel has set the power mode

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes   # tell systemd the service is "active" after the command runs

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jetson-clocks
sudo systemctl start jetson-clocks
```

### N.3.5 GPU Layer Offloading on Jetson

On a desktop PC with a discrete GPU, the decision of how many layers to offload (`-ngl`) is constrained by VRAM capacity — if you offload too many layers, they don't fit in VRAM and the build either fails or silently falls back to CPU. On Jetson (unified memory), there is no separate VRAM pool. Every layer you offload to the GPU is still in the same physical LPDDR5 — you are simply changing which compute units process it.

This means the `--n-gpu-layers` decision on Jetson is purely about **compute allocation**, not memory capacity. The GPU's CUDA cores are much faster at matrix multiply than the CPU cores for any model with more than a few billion parameters. You should almost always offload all layers:

```bash
# Full GPU acceleration — offload all layers to CUDA
# The '999' value is larger than any model has layers; excess is ignored
./build/bin/llama-server \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 4096 \
    --flash-attn \
    --parallel 2 \
    --cont-batching \
    --host 0.0.0.0 \
    --port 8080

# Verify GPU is doing the work by watching tegrastats during inference
tegrastats --interval 500
# Key field: GR3D_FREQ — this should be >60% during active generation
# GR3D is NVIDIA's internal name for the GPU's graphics/compute engine
# If GR3D_FREQ is near 0%, the GPU is idle and CPU is doing everything
```

**When to reduce `-ngl`:** Only if you are running the Jetson as a general-purpose computer with other GPU-intensive workloads running simultaneously. For a dedicated inference server, always use `-ngl 999`.

### N.3.6 Memory Budget and Quantization for Jetson

The memory budget calculation on Jetson is the same as any unified-memory system. The formula:

```
Available = Total_RAM − OS_overhead (~1.5 GB) − KV_cache − buffers (~0.5 GB)

KV_cache = 2 × num_layers × num_kv_heads × head_dim × ctx_size × bytes_per_element
         (for a 7B model, bf16 KV, 8K context:
          = 2 × 32 × 32 × 128 × 8192 × 2 ≈ 4.3 GB)
```

For Orin Nano 8 GB running 7B Q4_K_M at 8K context:

- OS: 1.5 GB, weights: 4.4 GB, KV cache: 4.3 GB, buffers: 0.5 GB → **10.7 GB total: doesn't fit**

At 4K context: KV cache drops to ~2.1 GB → **8.5 GB total: barely fits**

At 3K context: KV cache drops to ~1.6 GB → **8.0 GB total: fits with margin**

Practical recommendation for Orin Nano 8 GB + 7B Q4_K_M: `--ctx-size 3072`.

| Module | Memory | Model | Quant | ctx | Tokens/sec |
|---|---|---|---|---|---|
| Orin Nano 4GB | 4 GB | Llama-3.2-1B | Q8_0 | 2048 | 35–50 |
| Orin Nano 4GB | 4 GB | Llama-3.2-3B | Q4_K_M | 2048 | 20–30 |
| Orin Nano 8GB | 8 GB | Qwen2.5-7B | Q4_K_M | 3072 | 15–22 |
| Orin NX 16GB | 16 GB | Qwen2.5-14B | Q4_K_M | 4096 | 10–15 |
| AGX Orin 32GB | 32 GB | Llama-3.1-8B | F16 | 8192 | 30–40 |
| AGX Orin 64GB | 64 GB | Llama-3.3-70B | Q4_K_M | 4096 | 8–12 |

### N.3.7 Flash Attention on Jetson Orin

Flash Attention is the tiled attention algorithm described in Chapter 5. Instead of computing the full N×N attention score matrix — which grows quadratically with sequence length and requires O(N²) memory — Flash Attention computes attention in tiles that fit in the GPU's fast on-chip SRAM, reading and writing the larger HBM/LPDDR memory only once per tile pass. The memory footprint for attention drops from O(N²) to O(N), and for long contexts the reduced bandwidth cost also improves speed.

Jetson Orin's Ampere GPU (sm_87) supports the CUDA primitives required for Flash Attention. Enabling it expands the practical context length you can fit within the Orin Nano's 8 GB — by roughly 30–50% — because the peak memory during attention computation is no longer the bottleneck.

```bash
# Enable Flash Attention with the --flash-attn flag
./build/bin/llama-server \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 8192 \    # can use larger context with Flash Attention
    --flash-attn \       # enable tiled attention kernel
    --parallel 1 \
    --host 0.0.0.0 \
    --port 8080
```

**Do not use `--flash-attn` on the original Jetson Nano (Maxwell/sm_53).** The required CUDA tile operations are not available in CUDA 10.2 / Maxwell, and the flag will either be silently ignored or produce incorrect output.

### N.3.8 Monitoring with tegrastats and jtop

Understanding what your Jetson is doing during inference requires reading its monitoring tools. Unlike a desktop Linux system where `nvidia-smi` provides GPU information, Jetson uses different tools because its GPU is an integrated SoC component, not a discrete PCIe device.

**tegrastats** is NVIDIA's built-in monitoring tool for Jetson. It prints one line per interval showing CPU usage, GPU usage, memory, temperature, and power draw:

```bash
tegrastats --interval 500   # print every 500ms
# Example output:
# RAM 3842/7773MB (lfb 4x2MB) SWAP 0/3886MB
# CPU [78%@2015,65%@2015,71%@2015,68%@2015,72%@2015,70%@2015]
# EMC_FREQ 40% GR3D_FREQ 85% AO@44C GPU@56C tj@58C VDD_IN 12.3W
```

Reading the output:

- `RAM 3842/7773MB` — using 3.8 GB of 7.8 GB total (the rest is reserved by the OS for GPU and other hardware)
- `lfb 4x2MB` — largest free block is 4 contiguous 2 MB pages — low fragmentation is good
- `GR3D_FREQ 85%` — GPU is at 85% utilisation — inference is GPU-bound (ideal)
- `GPU@56C` — GPU temperature at 56°C — healthy
- `VDD_IN 12.3W` — total board power draw is 12.3 W

**jtop** is a community-maintained monitoring tool with a more readable terminal UI:

```bash
pip3 install jetson-stats
sudo jtop
# Navigate with arrow keys
# GPU tab: CUDA utilisation, memory allocation, power per component
# ALL tab: every sensor simultaneously — useful for thermal debugging
```

### N.3.9 Thermal Management on Jetson

Jetson modules generate significantly more heat than Raspberry Pi, proportional to their higher compute throughput. An AGX Orin running at 60W produces as much heat as a demanding desktop CPU.

**Temperature thresholds for all Orin modules:**

- Safe sustained: below 80°C
- Throttle onset: 85°C (clock speed reduction begins)
- Hard shutdown: 95°C (OS initiates immediate shutdown to protect hardware)

Unlike the Raspberry Pi, Jetson dev kits ship with appropriate active cooling included. The Orin Nano and NX dev kits include a fan-heatsink assembly. The AGX Orin dev kit includes a large fan. For production deployments using bare Jetson modules (without the dev kit carrier board), you must design and attach a custom heatsink — NVIDIA publishes thermal interface specifications in the System-on-Module Design Guide.

```bash
# Monitor thermal state during inference (run in a second terminal)
watch -n 1 "tegrastats 2>/dev/null | grep -oP 'GPU@\d+C|tj@\d+C|GR3D_FREQ \d+%'"

# Check for throttle events in the kernel log
sudo dmesg | grep -i "thermal\|throttl\|temperature"

# If you see throttle events, check the power mode is set correctly
sudo nvpmodel -q   # should show mode 0 (MAXN) for max performance
```

### N.3.10 Production Server Configuration for Jetson

```bash
#!/bin/bash
# /usr/local/bin/start-llama-server.sh
# This script locks clocks before starting inference and then launches the server.
# Using 'exec' ensures that signals (SIGTERM from systemd during shutdown)
# are delivered directly to llama-server, not to this shell script.

/usr/bin/jetson_clocks   # lock clocks at maximum before inference starts

exec /home/jetson/llama.cpp/build/bin/llama-server \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \        # all layers to GPU
    --ctx-size 4096 \
    --flash-attn \
    --parallel 4 \              # 4 concurrent request slots
    --cont-batching \           # continuous batching across slots
    --batch-size 512 \          # tokens processed per prefill step
    --ubatch-size 128 \         # micro-batch size for decode
    --mlock \                   # lock weights in RAM against swap
    --host 0.0.0.0 \
    --port 8080 \
    --metrics \                 # expose /metrics for Prometheus
    --log-prefix \
    --log-timestamps
```

```ini
# /etc/systemd/system/llama-server.service
[Unit]
Description=llama.cpp inference server (Jetson)
After=network.target nvpmodel.service
Requires=network.target

[Service]
Type=simple
User=jetson
ExecStartPre=/usr/bin/nvpmodel -m 0   # ensure MAXN mode before starting
ExecStart=/usr/local/bin/start-llama-server.sh
Restart=on-failure
RestartSec=10
LimitNOFILE=65536
LimitMEMLOCK=infinity   # required for --mlock to work as a non-root user

[Install]
WantedBy=multi-user.target
```

```bash
sudo chmod +x /usr/local/bin/start-llama-server.sh
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Verify
journalctl -u llama-server -f
curl http://localhost:8080/health
```

### N.3.11 TensorRT-LLM on Jetson AGX Orin

For the AGX Orin — the most powerful Jetson module — NVIDIA's TensorRT-LLM can deliver 2–3× higher throughput than llama.cpp for sustained multi-user workloads.

**What TensorRT-LLM does that llama.cpp does not:** llama.cpp optimises at the kernel level — each CUDA kernel is well-tuned for its specific operation. TensorRT-LLM optimises at the graph level — it analyses the entire model's computational graph, fuses adjacent operations into single kernels, and applies compiler-level optimisations across operation boundaries. It also applies INT4-AWQ quantization with calibration-aware compensation, which provides better quality than naively quantizing to INT4. The result is a compiled inference engine specific to your hardware and model, which cannot be shared across different GPU types but runs optimally on the specific GPU it was compiled for.

The trade-off is complexity: TensorRT-LLM requires converting models from Hugging Face format, running a compilation step that takes 10–30 minutes, and managing a separate engine artifact. For a production AGX Orin deployment where throughput matters, it is worth the setup cost.

```bash
pip3 install tensorrt-llm --extra-index-url https://pypi.ngc.nvidia.com

# Convert model and compile TRT-LLM engine
python3 -m tensorrt_llm.commands.build \
    --model_dir /mnt/nvme/hf_models/Qwen2.5-7B-Instruct \
    --output_dir /mnt/nvme/trt_engines/qwen2.5-7b-awq \
    --dtype float16 \
    --tp_size 1 \
    --max_batch_size 8 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --use_gpt_attention_plugin float16 \
    --use_gemm_plugin float16 \
    --quantization int4_awq   # activation-aware weight quantization

# Serve the compiled engine
python3 -m tensorrt_llm.serve \
    --engine_dir /mnt/nvme/trt_engines/qwen2.5-7b-awq \
    --tokenizer /mnt/nvme/hf_models/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Expected throughput on AGX Orin at 60W: 30–50 tokens/sec for a 7B INT4-AWQ model with continuous batching — roughly 2–3× the llama.cpp result on the same hardware.

### N.3.12 Benchmarking on Jetson

```bash
# Standard benchmark using llama-bench
./build/bin/llama-bench \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --flash-attn \
    -p 512 \    # prompt tokens — tests prefill (pp) throughput
    -n 128 \    # generation tokens — tests decode (tg) throughput
    -r 3        # repeat 3 times and report mean ± standard deviation

# Expected output on Orin Nano 8GB:
# pp512 = 280 t/s   (prefill throughput)
# tg128 = 18 t/s    (decode throughput — the number users experience)

# Run tegrastats during the benchmark to see GPU utilisation
tegrastats --interval 250 > /tmp/bench_tegrastats.log &
TEGRA_PID=$!
./build/bin/llama-bench ...
kill $TEGRA_PID

# Compute mean GPU utilisation during benchmark
grep "GR3D_FREQ" /tmp/bench_tegrastats.log | \
    grep -oP 'GR3D_FREQ \K\d+' | \
    awk '{sum+=$1; n++} END {printf "Mean GPU utilisation: %.1f%%\n", sum/n}'
```

If mean GPU utilisation is below 70% during the benchmark, something is limiting GPU throughput — check that `--n-gpu-layers 999` is set and that the CUDA build was compiled for the correct architecture (`sm_87` for Orin).

---

## N.4 Platform Comparison — Pi vs Jetson

| Dimension | Raspberry Pi 5 (8GB) | Jetson Orin Nano (8GB) | Jetson AGX Orin (64GB) |
|---|---|---|---|
| **GPU for inference** | Vulkan (experimental) | CUDA Ampere (full) | CUDA Ampere (full) |
| **7B Q4 decode speed** | 4–6 tok/s | 15–22 tok/s | 40–55 tok/s |
| **Max practical model** | 7B Q4_K_M | 7B Q8 / 14B Q4 | 70B Q4_K_M |
| **Context at 4K tokens** | Fits | Fits | Fits easily |
| **Flash Attention** | No (VideoCore) | Yes (Ampere) | Yes (Ampere) |
| **TensorRT-LLM** | No | Limited | Full support |
| **Power draw** | 5–12 W | 7–15 W | 15–60 W |
| **OS** | Raspberry Pi OS (Debian) | Ubuntu 22.04 (JetPack) | Ubuntu 22.04 (JetPack) |
| **Install complexity** | Low | Medium (JetPack required) | Medium |
| **Cost** | ~$80 + accessories | ~$199 dev kit | ~$499 dev kit |
| **Concurrent users** | 1 practical | 2–4 | 8–16 |
| **Best for** | Local/hobby/privacy | Edge AI products | Production edge |

---

## N.5 Choosing the Right Platform

**Choose Raspberry Pi 5 when:**

- Budget is the primary constraint and $80 is meaningful
- The use case is single-user and interactive latency is not critical (home automation, overnight batch jobs)
- You need broad Linux compatibility with minimal toolchain complexity
- The model is 3B parameters or smaller, or you can tolerate 4–6 tokens/sec on a 7B model
- Power consumption must stay under 12W

**Choose Jetson Orin Nano 8GB when:**

- You need real GPU-accelerated CUDA inference (15–22 tok/s for 7B)
- The application serves 2–4 concurrent users
- Flash Attention and cuBLAS optimisations matter for your workload
- You are building a product where the dev kit is a prototype path to the production SOM
- You can tolerate JetPack setup complexity (it is a one-time cost)

**Choose Jetson AGX Orin when:**

- 13B–70B model serving is required at the edge
- Production multi-user deployment (8–16 concurrent requests)
- TensorRT-LLM optimization is worth the additional build complexity
- The power budget allows 15–60W and the thermal environment can handle it

---

## N.6 Common Failure Modes and Fixes

| Symptom | Likely Cause | Diagnosis | Fix |
|---|---|---|---|
| OOM at load time | Model + KV cache exceeds RAM | Check `free -h` during load | Use smaller quant or reduce `--ctx-size` |
| 1–2 tok/s on Pi 5 | Thermal throttling | `vcgencmd get_throttled` returns nonzero | Improve cooling; check governor |
| Pi 4 running slowly | Wrong Pi model for LLMs | Expected on Pi 4 | Upgrade to Pi 5, or use smaller model |
| GR3D_FREQ near 0% on Jetson | CUDA not in build | `llama-cli --version` shows "CPU only" | Rebuild with `-DGGML_CUDA=ON` |
| `CUDA error: no kernel image` | Wrong CUDA architecture flag | Check `nvcc --version` and GPU model | Rebuild with correct `CMAKE_CUDA_ARCHITECTURES` |
| Model loads but output garbled | Architecture mismatch | Compare expected vs actual sm_ flag | Rebuild with correct flag |
| 45-second model load | Model on SD card | `dd` benchmark on storage | Move model to NVMe or USB SSD |
| Server killed after hours | OOM by OS | Check kernel log: `dmesg | grep oom` | Reduce `--ctx-size`; add swap |
| `nvpmodel` resets after reboot | Not persisted | `sudo nvpmodel -q` shows wrong mode | Enable jetson-clocks systemd service |
| Power supply brown-out | USB-C supply underrated | Pi: check `vcgencmd get_throttled` for bit 3 | Use official 27W supply; for Jetson: barrel connector |

---

## N.7 Quick-Reference Commands

```bash
# ── Raspberry Pi ─────────────────────────────────────────────────────

# Confirm 64-bit OS
uname -m               # must print aarch64

# Temperature and throttle check
vcgencmd measure_temp
vcgencmd get_throttled # 0x0 = healthy

# Set performance CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Run CPU-only server
llama-server --model MODEL.gguf --port 8080

# Run with Vulkan GPU offload (Pi 5 only — build with -DGGML_VULKAN=ON first)
llama-server --model MODEL.gguf --n-gpu-layers 20 --port 8080

# View systemd service logs
journalctl -u llama-server -f

# ── Jetson ───────────────────────────────────────────────────────────

# Check JetPack version
cat /etc/nv_tegra_release

# Set maximum performance mode and lock clocks
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor GPU utilisation and temperature
tegrastats --interval 500

# Full GPU inference
llama-server --model MODEL.gguf --n-gpu-layers 999 --flash-attn --port 8080

# Benchmark prefill and decode throughput
llama-bench --model MODEL.gguf --n-gpu-layers 999 -p 512 -n 128 -r 3

# Check GPU utilisation is high during inference
# Look for GR3D_FREQ > 70% in tegrastats output
```

---

*For Apple Silicon and Android deployment, see Appendix M. For the general llama.cpp CLI flag reference, see Appendix D. For CI/CD pipelines that build llama.cpp binaries for these platforms automatically, see Appendix O (the llama.cpp multi-architecture build matrix section).*
