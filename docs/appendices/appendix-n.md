# Appendix N — Edge Inference on Linux SBCs: Raspberry Pi and NVIDIA Jetson

> *"These are not toy deployments. A Raspberry Pi 5 can serve a quantized 7B model to a local home-automation stack. A Jetson Orin AGX can serve a 70B model with GPU acceleration in a 60-watt envelope. The physics is real; only the scale changes."*

---

## N.1 Platform Overview and Hardware Comparison

Both platforms run Linux and support llama.cpp natively. They make fundamentally different trade-offs.

| Property | Raspberry Pi 4 (8 GB) | Raspberry Pi 5 (8 GB) | Jetson Nano (4 GB) | Jetson Orin Nano (8 GB) | Jetson AGX Orin (64 GB) |
|---|---|---|---|---|---|
| CPU | Cortex-A72 4-core | Cortex-A76 4-core | Cortex-A57 4-core | Cortex-A78AE 6-core | Cortex-A78AE 12-core |
| GPU | VideoCore VI | VideoCore VII | Maxwell 128-core | Ampere 1024-core | Ampere 2048-core |
| GPU backend | Vulkan (experimental) | Vulkan | CUDA 10.2 | CUDA 11.4 | CUDA 11.4 |
| Memory | 8 GB LPDDR4X shared | 8 GB LPDDR4X shared | 4 GB LPDDR4 shared | 8 GB LPDDR5 shared | 64 GB LPDDR5 shared |
| Memory BW | ~25 GB/s | ~51 GB/s | ~25.6 GB/s | ~68 GB/s | ~204 GB/s |
| TDP | 5–7 W | 5–12 W | 5–10 W | 7–15 W | 15–60 W |
| Price (approx) | $80 | $80 | $149 (discontinued) | $199 | $499 |
| Best for | CPU-only edge serving | CPU + light Vulkan | Legacy CUDA projects | Production edge GPU | Serious edge deployment |

**The critical insight:** On all of these platforms, CPU and GPU share the same physical memory pool. There is no PCIe bus transfer cost when the GPU reads a weight that the CPU loaded. This is the same unified-memory advantage described in Appendix M for Apple Silicon — except that bandwidth is typically lower and thermal headroom is tighter.

---

## N.2 Part One: Raspberry Pi

### N.2.1 Which Pi Should You Use?

**Pi 5 (8 GB) — the only Pi worth running LLMs on in 2026.** The Cortex-A76 cores are roughly 2–3× faster than the A72 in Pi 4 for the matrix-multiply patterns that dominate LLM decode. The Pi 4 will run models but is frustratingly slow — expect 1–2 tokens/sec for a Q4 7B model. The Pi 5 does 4–8 tokens/sec for the same model, which is usable for local tooling.

The **Pi Zero 2W (512 MB)** cannot run any model of practical size. The **Pi CM4** is the same silicon as Pi 4; the compute module form factor is useful for integration but does not change performance.

**Minimum viable setup:** Pi 5 8 GB + active cooling + USB 3 NVMe HAT (for fast model loading) + official 27W USB-C power supply.

### N.2.2 OS and Initial Setup

Use the 64-bit Raspberry Pi OS (Bookworm, 2024 or later). The 32-bit image caps usable RAM at ~3 GB and halves NEON SIMD register width — do not use it for inference.

```bash
# Verify you are on a 64-bit kernel
uname -m   # must print aarch64, not armv7l

# Update everything before building
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y cmake git build-essential python3-dev \
    libopenblas-dev pkg-config
```

**Swap configuration.** llama.cpp loads models into RAM. A Q4_K_M 7B model is ~4.1 GB. On an 8 GB Pi you have ~7 GB available after the OS, which is enough — but without swap, any memory spike will OOM-kill the process. Set 4 GB of swap on a fast storage device (not the SD card):

```bash
# If you have an NVMe or USB SSD mounted at /mnt/nvme:
sudo fallocate -l 4G /mnt/nvme/swapfile
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile

# Make it persist across reboots
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Reduce swappiness so swap is only used under real pressure
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### N.2.3 Building llama.cpp on Raspberry Pi

The standard CPU build uses OpenBLAS for batched matrix operations and NEON SIMD intrinsics for the hot path.

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CPU-only build with OpenBLAS
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_NATIVE=ON

cmake --build build --config Release -j4
```

`-DGGML_NATIVE=ON` enables compiler auto-vectorization tuned to the exact CPU — this matters on ARM where NEON/SVE availability varies by silicon revision.

**Build time:** approximately 8–12 minutes on a Pi 5. Let it complete fully; partial builds produce silently wrong binaries.

**Verify the build:**

```bash
./build/bin/llama-cli --version
# Should print version string and mention: CPU, BLAS
```

### N.2.4 Vulkan Backend (Pi 5 Only)

The Pi 5's VideoCore VII supports Vulkan 1.2. llama.cpp can offload matrix multiplications to the GPU through the Vulkan backend. The VideoCore VII is not a high-throughput GPU — do not expect Apple Silicon-level speedups — but offloading a moderate number of layers (8–16 out of 32 for a 7B model) reduces CPU thermal load and can improve sustained throughput.

```bash
# Install Vulkan development libraries
sudo apt install -y libvulkan-dev vulkan-tools glslc

# Confirm Vulkan is available
vulkaninfo --summary 2>/dev/null | grep "GPU id"
# Should list: Broadcom Limited V3D 7.1.5

# Build with Vulkan backend
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON

cmake --build build --config Release -j4
```

**Running with Vulkan layer offloading:**

```bash
# Offload 16 layers to VideoCore VII GPU
./build/bin/llama-server \
    --model /mnt/nvme/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 16 \
    --ctx-size 2048 \
    --host 0.0.0.0 \
    --port 8080
```

**Calibrating `-n-gpu-layers` on Pi 5:** Start at 0 and increase by 4 until you see `GGML_VULKAN: Allocated` in the startup log. If the process OOMs, reduce by 4. The VideoCore VII has access to the full 8 GB pool, but allocating too many layers leaves the CPU without enough RAM for the KV cache.

A practical rule: for a Q4_K_M 7B model on Pi 5 8 GB, `-n-gpu-layers 20` offloads the compute-heavy attention layers while leaving ~3 GB for the OS and KV cache.

### N.2.5 Model Storage: SD Card vs. NVMe

A model loaded from a Class 10 SD card takes 45–90 seconds to reach the first token. The same model from a USB 3.0 NVMe enclosure (or the Pi 5's PCIe M.2 HAT) loads in 8–15 seconds. For any interactive application this difference matters.

```bash
# Benchmark your storage device
dd if=/mnt/nvme/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
   of=/dev/null bs=1M status=progress
# Target: >200 MB/s for NVMe, >80 MB/s for good USB SSD
# SD cards typically deliver 20-40 MB/s sequential read
```

The Pi 5 PCIe interface supports Gen 2 (but not Gen 3 by default). Enable Gen 3 in `/boot/firmware/config.txt`:

```ini
# Add to /boot/firmware/config.txt
dtparam=pciex1_gen=3
```

Reboot and verify with `lspci -vv | grep -i speed`.

### N.2.6 Quantization Tiers for Raspberry Pi

| Available RAM | Model | Recommended Quant | File Size | Tokens/sec (Pi 5) |
|---|---|---|---|---|
| 8 GB | Llama-3.2-3B | Q8_0 | 3.3 GB | 7–9 |
| 8 GB | Llama-3.2-3B | Q4_K_M | 1.9 GB | 9–12 |
| 8 GB | Qwen2.5-7B | Q4_K_M | 4.4 GB | 4–6 |
| 8 GB | Qwen2.5-7B | Q2_K | 2.7 GB | 5–8 |
| 8 GB | Llama-3.1-8B | Q4_K_M | 4.9 GB | 3–5 |
| 4 GB | Llama-3.2-1B | Q4_K_M | 0.8 GB | 12–18 |
| 4 GB | Phi-3-mini (3.8B) | Q4_K_M | 2.2 GB | 6–9 |

**Do not attempt 13B+ models on 8 GB Pi.** Even Q2_K quantization pushes 13B to ~5.2 GB weights alone, leaving insufficient room for the OS, KV cache, and inference buffers.

**Q4_K_M is the sweet spot** for Pi 5. It uses 4-bit weights with k-quant mixed precision (some layers at higher precision) and delivers better perplexity than plain Q4_0 at similar throughput.

### N.2.7 Thermal Management

The Pi 5 throttles at 85°C and hard-shuts at 90°C. Sustained LLM inference is one of the most thermally intensive workloads a Pi will encounter.

**Cooling requirements:**
- **Passive heatsink only:** Not sufficient for sustained inference. Core temperature reaches throttle limit within 5–10 minutes.
- **Official Pi 5 Active Cooler (Pimoroni/official):** Keeps core below 75°C during sustained inference. This is the minimum viable solution.
- **Fan + heatsink case (Argon NEO 5, etc.):** Keeps core at 60–70°C. Recommended for always-on deployments.

**Monitor temperature while running:**

```bash
# Real-time temperature monitoring
watch -n 2 "vcgencmd measure_temp && vcgencmd get_throttled"

# Decode the throttled bitmask (0x0 = no throttling)
# 0x50005 = currently throttled + previously throttled + freq cap
```

If `get_throttled` returns nonzero during inference, your cooling is insufficient. Throttled inference is self-defeating: performance drops to the same level as a cooler Pi running at full clock.

**Increase CPU governor to performance mode:**

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

### N.2.8 Running llama-server as a systemd Service

For a persistent local inference endpoint that survives reboots:

```ini
# /etc/systemd/system/llama-server.service
[Unit]
Description=llama.cpp inference server
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
ExecStart=/home/pi/llama.cpp/build/bin/llama-server \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --ctx-size 4096 \
    --n-gpu-layers 20 \
    --parallel 1 \
    --host 0.0.0.0 \
    --port 8080 \
    --log-disable
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Verify it's running
sudo systemctl status llama-server
curl http://localhost:8080/health
```

### N.2.9 Realistic Performance Expectations

A Pi 5 running a Q4_K_M 7B model will deliver 4–6 tokens/sec in decode under normal thermal conditions. This is enough for:

- Local chatbot serving a single user
- Home-automation natural-language command parsing
- Batch document summarization (where latency is not interactive)
- Private, offline text processing

It is not enough for multi-user production serving, real-time transcription, or anything requiring sub-second time-to-first-token on long prompts.

For multi-user local serving on ARM without CUDA, consider the **Raspberry Pi 5 Cluster** pattern: 4× Pi 5 behind an nginx upstream, each handling one concurrent request, aggregate throughput ~20–24 tokens/sec, total cost ~$400.

---

## N.3 Part Two: NVIDIA Jetson

### N.3.1 The Jetson Product Line in 2026

NVIDIA's Jetson family spans a wide capability range. For LLM inference the relevant products are:

| Module | GPU | CUDA Cores | Memory | BW | TDP | Notes |
|---|---|---|---|---|---|---|
| Jetson Nano (2019) | Maxwell | 128 | 4 GB LPDDR4 | 25.6 GB/s | 5–10 W | Discontinued; Maxwell CUDA 10.2 only |
| Jetson Nano 2GB | Maxwell | 128 | 2 GB LPDDR4 | 25.6 GB/s | 5–10 W | Too little RAM for useful LLMs |
| Jetson Orin Nano 4GB | Ampere | 512 | 4 GB LPDDR5 | 34 GB/s | 7–10 W | Entry Orin; tight on memory |
| Jetson Orin Nano 8GB | Ampere | 1024 | 8 GB LPDDR5 | 68 GB/s | 7–15 W | **Recommended entry point** |
| Jetson Orin NX 8GB | Ampere | 1024 | 8 GB LPDDR5 | 102 GB/s | 10–20 W | Higher BW than Orin Nano |
| Jetson Orin NX 16GB | Ampere | 1024 | 16 GB LPDDR5 | 102 GB/s | 10–25 W | Comfortable for 7B at Q8 |
| Jetson AGX Orin 32GB | Ampere | 2048 | 32 GB LPDDR5 | 204 GB/s | 15–60 W | 13B models at full precision |
| Jetson AGX Orin 64GB | Ampere | 2048 | 64 GB LPDDR5 | 204 GB/s | 15–60 W | 70B at Q4_K_M |

**The original Jetson Nano (Maxwell) is a special case.** It supports CUDA 10.2 only, which is too old for many modern llama.cpp CUDA optimizations. It can still run llama.cpp in CPU mode or with limited CUDA offloading, but its performance is comparable to a Raspberry Pi 5 — without the Pi 5's lower cost. For new projects, the **Jetson Orin Nano 8GB** is the correct entry point.

### N.3.2 JetPack SDK and System Setup

All Jetson platforms require **JetPack** — NVIDIA's board support package that bundles the OS (Ubuntu-based), CUDA toolkit, cuDNN, TensorRT, and hardware drivers. Do not try to install a generic Ubuntu on a Jetson; the GPU and CSI interfaces require NVIDIA's patched kernel.

**Check your JetPack version:**

```bash
cat /etc/nv_tegra_release
# or
dpkg -l | grep jetpack
```

For llama.cpp CUDA inference, you need **JetPack 5.x or 6.x** (Orin series ships with JetPack 5.1 or 6.0). The original Nano ships with JetPack 4.6, which provides CUDA 10.2.

**Flash JetPack using SDK Manager** (run on a host x86_64 Ubuntu machine connected via USB):

```bash
# On host machine — install NVIDIA SDK Manager
# https://developer.nvidia.com/sdk-manager
sdkmanager --cli install \
    --product Jetson \
    --target JETSON_ORIN_NANO_8GB \
    --version 6.0 \
    --select-default
```

**After flashing — expand the rootfs to your NVMe:**

The eMMC or SD card on Jetson modules is typically 16–32 GB — too small for large models. Mount an NVMe SSD and either:

1. Store models on NVMe and keep rootfs on eMMC
2. Move the entire rootfs to NVMe (preferred for development boards)

```bash
# Store models on NVMe mounted at /mnt/nvme
sudo mkdir -p /mnt/nvme
# Add to /etc/fstab using UUID from blkid
echo "UUID=$(blkid /dev/nvme0n1p1 -s UUID -o value) \
    /mnt/nvme ext4 defaults 0 2" | sudo tee -a /etc/fstab
sudo mount -a
```

### N.3.3 Building llama.cpp with CUDA on Jetson

The key difference from desktop CUDA builds: Jetson uses the **Tegra CUDA library path** and the ARM64 compiler toolchain. The cmake invocation must account for this.

**Install build dependencies:**

```bash
sudo apt update
sudo apt install -y cmake git build-essential python3-pip \
    libopenblas-dev pkg-config ninja-build

# Verify CUDA is available
nvcc --version
nvidia-smi   # Note: nvidia-smi works differently on Jetson
             # Use tegrastats instead for power/thermal monitoring
```

**Clone and build with CUDA backend:**

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CUDA build — CUDA_ARCHITECTURES depends on your module
# Jetson Nano (Maxwell):    sm_53
# Jetson Orin (Ampere):     sm_87
# Jetson AGX Orin (Ampere): sm_87

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_NATIVE=ON

cmake --build build --config Release -j$(nproc)
```

**Build time:** 15–25 minutes on Orin Nano. The CUDA kernels are the bottleneck.

**Verify CUDA is active:**

```bash
./build/bin/llama-cli --version
# Should include: CUDA
# Should NOT say: CPU only

# Quick smoke test
./build/bin/llama-cli \
    --model /mnt/nvme/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 99 \
    -p "The capital of France is" \
    -n 20
# Watch for: "CUDA0: ..." allocation messages during load
```

**For the original Jetson Nano (CUDA 10.2 / sm_53):**

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=53 \
    -DGGML_NATIVE=ON

cmake --build build --config Release -j4
```

Note that CUDA 10.2 does not support FlashAttention. The `--flash-attn` flag will have no effect and may produce a warning.

### N.3.4 Power Mode Configuration

Jetson boards have configurable power modes that set CPU frequency caps and GPU power budgets. Inference performance is directly tied to the active power mode.

```bash
# List available power modes
sudo nvpmodel -q --verbose

# Common mode IDs (vary by board — check your board's documentation)
# Jetson Orin Nano:
#   Mode 0: MAXN (maximum performance, ~15W)
#   Mode 1: 10W
#   Mode 2: 7W

# Set to maximum performance mode
sudo nvpmodel -m 0

# Lock CPU and GPU clocks at maximum frequency
sudo jetson_clocks

# Verify
sudo jetson_clocks --show
```

**Important:** `jetson_clocks` does not persist across reboots. Add it to `/etc/rc.local` or a systemd service for permanent effect:

```bash
# /etc/systemd/system/jetson-clocks.service
[Unit]
Description=Lock Jetson clocks to maximum
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable jetson-clocks
sudo systemctl start jetson-clocks
```

### N.3.5 GPU Layer Offloading Strategy

On Jetson, all memory is unified — offloading layers to the GPU does not consume a separate GPU memory pool; it uses the same physical LPDDR5. The decision of how many layers to offload is therefore about **compute throughput** (GPU CUDA cores vs. CPU cores), not memory capacity.

**General principle:** Offload as many layers as possible (`-n-gpu-layers 999` or `-ngl 999`). The GPU's CUDA cores will outperform the CPU cores for matrix-multiply on any Orin-family device. The only reason to reduce `-ngl` is if you need to leave compute capacity for the CPU to handle other workloads.

```bash
# Fully GPU-accelerated inference on Orin
./build/bin/llama-server \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 8192 \
    --flash-attn \
    --parallel 2 \
    --cont-batching \
    --host 0.0.0.0 \
    --port 8080
```

**Verify GPU is doing the work** by watching tegrastats during inference:

```bash
tegrastats --interval 500
# Look for: GR3D_FREQ field — should show >60% during token generation
# GR3D is Jetson's name for the GPU compute engine
```

If GR3D_FREQ is near 0% during decode, CUDA is not being used — check that the build included CUDA and that `-ngl` is set.

### N.3.6 Quantization Tiers for Jetson

| Module | Memory | Model | Quant | File Size | Tokens/sec |
|---|---|---|---|---|---|
| Orin Nano 4GB | 4 GB | Llama-3.2-1B | Q8_0 | 1.5 GB | 35–50 |
| Orin Nano 4GB | 4 GB | Llama-3.2-3B | Q4_K_M | 1.9 GB | 20–30 |
| Orin Nano 8GB | 8 GB | Qwen2.5-7B | Q4_K_M | 4.4 GB | 15–22 |
| Orin Nano 8GB | 8 GB | Qwen2.5-7B | Q8_0 | 7.7 GB | 10–14 |
| Orin NX 16GB | 16 GB | Llama-3.1-8B | Q8_0 | 8.5 GB | 18–25 |
| Orin NX 16GB | 16 GB | Qwen2.5-14B | Q4_K_M | 8.9 GB | 10–15 |
| AGX Orin 32GB | 32 GB | Llama-3.1-8B | F16 | 16 GB | 30–40 |
| AGX Orin 32GB | 32 GB | Llama-3.3-70B | Q2_K | 26 GB | 5–8 |
| AGX Orin 64GB | 64 GB | Llama-3.3-70B | Q4_K_M | 41 GB | 8–12 |
| AGX Orin 64GB | 64 GB | Llama-3.3-70B | Q8_0 | 70 GB | Doesn't fit |

**Memory budget calculation for Jetson:**

```
Available for model = Total_RAM - OS (~1.5 GB) - KV_cache - inference_buffers (~0.5 GB)
KV_cache = 2 × num_layers × num_heads × head_dim × ctx_size × bytes_per_element
         = 2 × 32 × 32 × 128 × 8192 × 2   (7B model, bf16 cache, 8K ctx)
         = ~4.3 GB
```

For an Orin Nano 8 GB running a 7B model at 8K context:
- OS: 1.5 GB
- KV cache: 4.3 GB
- Weights: 4.4 GB (Q4_K_M)
- Total: 10.2 GB — **does not fit**

Reduce context to 4K:
- KV cache: 2.1 GB
- Total: 8.0 GB — **fits with ~0 margin**

Practical recommendation for Orin Nano 8 GB + 7B Q4_K_M: `--ctx-size 3072`. This leaves a safe buffer.

### N.3.7 Flash Attention on Jetson Orin

Jetson Orin (Ampere, sm_87) supports Flash Attention through llama.cpp's `--flash-attn` flag. This reduces KV cache memory bandwidth by computing attention in tiles without materializing the full attention score matrix — the same algorithm described in Chapter 5.

```bash
./build/bin/llama-server \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 8192 \
    --flash-attn \          # enable FlashAttention kernel
    --parallel 1 \
    --host 0.0.0.0 \
    --port 8080
```

With `--flash-attn`, the effective context you can fit at a given memory budget increases by roughly 30–50% on Orin hardware, because the peak memory during the attention computation is O(n) rather than O(n²).

**Note:** Do not use `--flash-attn` on the original Jetson Nano (CUDA 10.2 / Maxwell). The required CUDA kernels are not available and the flag will be silently ignored or produce incorrect output.

### N.3.8 Monitoring with tegrastats and jtop

```bash
# tegrastats — NVIDIA's built-in monitoring tool
tegrastats
# Output format:
# RAM 2915/7773MB (lfb 10x2MB) SWAP 0/3886MB CPU [45%@1984,38%@1984,...] \
# EMC_FREQ 40% GR3D_FREQ 78% AO@44C GPU@52C tj@53C

# Key fields:
# RAM — used/total, lfb = largest free block (fragmentation indicator)
# GR3D_FREQ — GPU utilization percentage
# GPU@52C — GPU temperature
# tj@53C — junction (die) temperature

# jtop — more readable third-party monitor
pip3 install jetson-stats
sudo jtop
```

**In jtop**, navigate with arrow keys. The GPU tab shows CUDA utilization, memory allocation, and power draw per component. The ALL tab shows every sensor simultaneously — useful for diagnosing thermal throttling.

### N.3.9 Thermal Management on Jetson

Jetson modules generate significantly more heat than Raspberry Pi. The AGX Orin at 60W requires a heatsink comparable to a desktop CPU cooler.

**Thermal targets:**

| Module | Max Safe Sustained Temp | Throttle Temp | Shutdown Temp |
|---|---|---|---|
| Orin Nano | 80°C | 85°C | 95°C |
| Orin NX | 80°C | 85°C | 95°C |
| AGX Orin | 80°C | 85°C | 95°C |

**Recommended cooling:**

- **Orin Nano / Nano Developer Kit:** Heatsink + 40mm fan (included in dev kit). Sufficient for sustained inference at 15W mode.
- **AGX Orin Developer Kit:** Active cooling with large heatsink fan included. Sufficient up to 40W. For 60W sustained workloads, add supplemental case airflow.
- **Production modules (without dev kit carrier):** Custom heatsink required. NVIDIA provides thermal interface specifications in the System-on-Module (SOM) Design Guide.

**Monitor and respond to thermal events:**

```bash
# Watch for throttling in real time
watch -n 1 "tegrastats 2>/dev/null | grep -oP 'tj@\d+C|GR3D_FREQ \d+%'"

# Check throttling events in kernel log
sudo dmesg | grep -i "thermal\|throttl"
```

### N.3.10 Production Server Configuration

A production llama-server on Jetson Orin with multiple concurrent users:

```bash
#!/bin/bash
# /usr/local/bin/start-llama-server.sh

# Lock clocks before starting inference
/usr/bin/jetson_clocks

# Start server
exec /home/jetson/llama.cpp/build/bin/llama-server \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 4096 \
    --flash-attn \
    --parallel 4 \
    --cont-batching \
    --batch-size 512 \
    --ubatch-size 128 \
    --mlock \
    --host 0.0.0.0 \
    --port 8080 \
    --metrics \
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
ExecStartPre=/usr/bin/nvpmodel -m 0
ExecStart=/usr/local/bin/start-llama-server.sh
Restart=on-failure
RestartSec=10
LimitNOFILE=65536
# Ensure full memory access
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
```

```bash
sudo chmod +x /usr/local/bin/start-llama-server.sh
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Check logs
journalctl -u llama-server -f
```

### N.3.11 Using TensorRT-LLM on Jetson AGX Orin

For the AGX Orin, NVIDIA's TensorRT-LLM (described in Chapter 33) provides higher throughput than llama.cpp for sustained multi-user workloads, because it applies graph-level optimization and operator fusion across the entire model graph rather than per-kernel optimization.

```bash
# Install TensorRT-LLM for Jetson (JetPack 6.x)
pip3 install tensorrt-llm --extra-index-url \
    https://pypi.ngc.nvidia.com

# Convert a Hugging Face model to TRT-LLM engine
python3 -m tensorrt_llm.commands.build \
    --model_dir /mnt/nvme/hf_models/Qwen2.5-7B-Instruct \
    --output_dir /mnt/nvme/trt_engines/qwen2.5-7b-q4 \
    --dtype float16 \
    --tp_size 1 \
    --max_batch_size 8 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --use_gpt_attention_plugin float16 \
    --use_gemm_plugin float16 \
    --quantization int4_awq

# Run the TRT-LLM server
python3 -m tensorrt_llm.serve \
    --engine_dir /mnt/nvme/trt_engines/qwen2.5-7b-q4 \
    --tokenizer /mnt/nvme/hf_models/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

TRT-LLM on AGX Orin at 60W can deliver 30–50 tokens/sec for a 7B INT4-AWQ model with continuous batching — approximately 2–3× the throughput of llama.cpp on the same hardware, at the cost of a more complex build and deployment pipeline.

### N.3.12 Benchmarking on Jetson

```bash
# llama-bench — standardized throughput benchmark
./build/bin/llama-bench \
    --model /mnt/nvme/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --flash-attn \
    -p 512 \     # prompt tokens (prefill)
    -n 128 \     # generation tokens (decode)
    -r 3          # repeat 3 times for stable measurement

# Expected output on Orin Nano 8GB:
# pp512 = 280 t/s   (prefill throughput)
# tg128 = 18 t/s    (decode throughput — the number users experience)

# Monitor GPU and power during benchmark
tegrastats --interval 250 > /tmp/bench_tegrastats.log &
TEGRA_PID=$!
./build/bin/llama-bench ...
kill $TEGRA_PID

# Analyze
grep "GR3D_FREQ" /tmp/bench_tegrastats.log | \
    awk '{sum+=$2; n++} END {print "Mean GR3D:", sum/n "%"}'
```

---

## N.4 Cross-Platform Comparison

| Dimension | Raspberry Pi 5 (8GB) | Jetson Orin Nano (8GB) | Jetson AGX Orin (64GB) |
|---|---|---|---|
| **GPU inference** | Vulkan (experimental) | CUDA (Ampere) | CUDA (Ampere) |
| **7B Q4 decode speed** | 4–6 tok/s | 15–22 tok/s | 40–55 tok/s |
| **Max practical model** | 7B Q4_K_M | 7B Q8 / 14B Q4 | 70B Q4_K_M |
| **Context at 8 GB** | 3K (7B Q4) | 3K (7B Q4) | N/A (more RAM) |
| **Power draw** | 5–12 W | 7–15 W | 15–60 W |
| **OS** | Raspberry Pi OS (Debian) | Ubuntu 22.04 (JetPack) | Ubuntu 22.04 (JetPack) |
| **Install complexity** | Low — standard cmake | Medium — JetPack required | Medium — JetPack required |
| **Cost** | ~$80 | ~$199 | ~$499 |
| **Best for** | Local/hobby inference | Edge AI applications | Production edge serving |
| **TensorRT-LLM** | No | Limited | Yes |
| **Flash Attention** | No (VideoCore) | Yes (Ampere) | Yes (Ampere) |
| **Multi-user serving** | 1 user practical | 2–4 users | 8–16 users |

---

## N.5 Choosing Between Pi and Jetson

**Choose Raspberry Pi 5 if:**
- Budget is the primary constraint (~$80 vs $199+)
- The use case is single-user local inference (chatbot, home automation)
- You need broad OS compatibility and simple tooling
- Your model is 3B parameters or smaller at Q4_K_M
- Power consumption must stay under 10W

**Choose Jetson Orin Nano 8GB if:**
- You need GPU-accelerated CUDA inference
- The model is 7B parameters and you need 15+ tokens/sec
- The application serves 2–4 concurrent users
- You are building a product (the developer kit is a path to the production SOM)
- Flash Attention and cuBLAS optimizations matter

**Choose Jetson AGX Orin if:**
- 13B–70B model serving is required
- Multi-user production deployment at the edge
- TensorRT-LLM optimization is worth the complexity
- You can tolerate 15–60W power draw and the associated thermal requirements

---

## N.6 Common Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| OOM at load time | Model too large for available RAM | Use smaller quant or reduce `--ctx-size` |
| 1–2 tok/s on Pi 5 | Thermal throttling | Check `vcgencmd get_throttled`; improve cooling |
| GR3D_FREQ = 0% on Jetson | CUDA not in build | Rebuild with `-DGGML_CUDA=ON`; verify `nvcc` available |
| Model loads but output is garbage | Wrong CUDA arch flag | Rebuild with correct `CMAKE_CUDA_ARCHITECTURES` (87 for Orin, 53 for Nano) |
| `CUDA error: no kernel image` | Architecture mismatch | Same as above |
| Server crashes after hours | Memory leak in long sessions | Add `--ctx-size` limit; set `Restart=on-failure` in systemd |
| 45-second model load time | Model on SD card | Move model to NVMe or USB SSD |
| Power supply brown-out | USB-C supply underrated | Use official 27W Pi supply; for Jetson use barrel-jack supply |
| jetson_clocks resets after reboot | Not in systemd | Enable `jetson-clocks.service` as shown in N.3.4 |

---

## N.7 Quick-Reference Commands

```bash
# ── Raspberry Pi ─────────────────────────────────────────────────────
# Check if 64-bit OS
uname -m

# Temperature and throttle check
vcgencmd measure_temp && vcgencmd get_throttled

# Set performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Run server with Vulkan (Pi 5 only)
llama-server --model MODEL.gguf --n-gpu-layers 20 --port 8080

# ── Jetson ───────────────────────────────────────────────────────────
# Check JetPack version
cat /etc/nv_tegra_release

# Set maximum performance mode
sudo nvpmodel -m 0 && sudo jetson_clocks

# Real-time monitoring
tegrastats --interval 500

# Run server with full GPU offload
llama-server --model MODEL.gguf --n-gpu-layers 999 --flash-attn --port 8080

# Benchmark
llama-bench --model MODEL.gguf --n-gpu-layers 999 -p 512 -n 128 -r 3
```

---

*For the general llama.cpp CLI flag reference, see Appendix D. For Apple Silicon and Android deployment, see Appendix M. For multi-GPU parallelism on data center hardware, see Chapter 15.*
