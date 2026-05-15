# Appendix M — Mobile and Edge Deployment: Android and Apple Silicon

> *"The same model weights, the same GGUF format, the same attention mechanism — running on a phone in your pocket or a laptop on your desk. The inference engine is the only thing that changes."*

---

## Why This Appendix Exists

Chapters 1–13 assume a data-center context: CUDA GPUs, HBM memory, Linux. But llama.cpp was explicitly designed to run anywhere a C++ compiler can reach. This appendix covers two of the most important non-data-center targets: **Android** (ARM and x86-64 mobile silicon) and **Apple Silicon** (the M-series unified-memory architecture). Both platforms present genuinely different constraints — battery, thermal throttling, unified memory pools, OS-level memory limits — that require a different mental model from server deployment.

**What you will know by the end:**

- Three distinct paths to running llama.cpp on Android: Android Studio GUI binding, Termux CLI, and NDK cross-compilation.
- How Android's SME2 (Arm) and AMX (x86-64) acceleration work and how llama.cpp auto-detects them.
- Apple Silicon's unified memory architecture and why it changes the quantization calculus.
- Metal GPU acceleration: build flags, layer offloading, and the difference between GPU-only and CPU+GPU split inference.
- Quantization tier recommendations by device class for both platforms.
- Memory budgeting, thermal management, and production-grade `llama-server` configuration for both.

---

## Part 1 — Android

### M.1 Platform Overview

Android inference runs on three distinct execution environments:

```
Android inference paths:
  ┌──────────────────────────────────────────────────────┐
  │ 1. Android Studio GUI binding (examples/llama.android)│
  │    — Kotlin/Java app, uses JNI bridge to llama.cpp   │
  │    — Hardware accel: SME2 (Arm), AMX (x86-64)        │
  │    — Best for: production apps, Play Store            │
  ├──────────────────────────────────────────────────────┤
  │ 2. Termux CLI                                        │
  │    — Full Linux-style shell on device, no root       │
  │    — Builds natively from source inside Termux       │
  │    — Best for: development, testing, prototyping     │
  ├──────────────────────────────────────────────────────┤
  │ 3. NDK cross-compilation                            │
  │    — Build on host, deploy via adb                   │
  │    — Fine-grained control over ABI and march flags   │
  │    — Best for: CI/CD pipelines, embedded products    │
  └──────────────────────────────────────────────────────┘
```

Hardware acceleration tiers (auto-detected at runtime):

| CPU feature | Devices | Speedup vs baseline |
|---|---|---|
| NEON (baseline) | All ARM64 since Android 5 | 1× |
| dotprod | Most devices since 2019 | ~1.5× |
| i8mm | Flagship devices since 2021 | ~2× |
| SVE2 | Cortex-X3/X4, some Snapdragon | ~2.5× |
| SME2 | Cortex-X925 (2024+) | ~3–4× |
| AMX | Intel Core Ultra (x86-64 Android/ChromeOS) | ~3× |

llama.cpp performs CPUID/HWCAP detection at startup and loads the appropriate kernel without user intervention.

---

### M.2 Path 1 — Android Studio GUI Binding

#### Prerequisites

```
Android Studio (latest stable)
Android NDK r27 or later (install via SDK Manager → SDK Tools)
CMake 3.22+ (install via SDK Manager → SDK Tools)
API level 28+ target (Android 9 minimum)
```

#### Import and Build

```
1. Open Android Studio
2. File → Open → navigate to llama.cpp/examples/llama.android
3. Wait for Gradle sync to complete
4. Build → Make Project  (or press Ctrl+F9 / ⌘F9)
```

The Gradle build invokes CMake internally and produces device-specific ABI splits (`arm64-v8a`, `x86_64`). The resulting APK is around 15–25 MB before model bundling.

#### Core API Surface

The binding exposes three primary objects:

**1. Parse model metadata without loading weights:**

```kotlin
// From a shared-storage Uri (user file picker)
val metadata = GgufMetadataReader.read(contentResolver, uri)
println("Architecture : ${metadata.architecture}")
println("Context size : ${metadata.contextLength}")
println("Parameter cnt: ${metadata.parameterCount}")
println("Quantization : ${metadata.quantizationType}")

// From a private app file
val metadata = GgufMetadataReader.read(File(filesDir, "model.gguf"))
```

This lets you display model information in your UI before committing to a load, which is important for guiding users who may have downloaded incompatible quantization tiers for their device RAM.

**2. Load a model and create an inference engine:**

```kotlin
// AiChat is a facade — manages lifecycle of the native context
val engine: InferenceEngine = AiChat.create(
    modelPath = File(filesDir, "model.gguf").absolutePath,
    contextSize = 4096,         // tokens; keep ≤ available RAM / bytes-per-token
    nThreads = 4,               // physical cores; avoid hyperthreads on mobile
    nGpuLayers = 0              // set > 0 if your device has Vulkan + llama.cpp GPU support
)
```

**3. Run inference and collect tokens as a Flow:**

```kotlin
val userPrompt = "Explain the transformer architecture in two sentences."

engine.generate(userPrompt)
    .collect { token ->
        // token is a String fragment — typically one subword
        textView.append(token)
    }

// Full example with lifecycle awareness
lifecycleScope.launch {
    engine.generate(userPrompt)
        .flowOn(Dispatchers.IO)
        .collect { fragment ->
            withContext(Dispatchers.Main) { outputTextView.append(fragment) }
        }
}
```

The `generate` call handles: chat template formatting (based on model metadata), prefill, KV cache management, and autoregressive decode. Cancelling the coroutine scope stops generation cleanly.

#### Choosing a Context Size

Mobile devices have hard per-process memory limits (typically 512 MB on low-end, 2–4 GB on flagship). Use this budget formula:

```
Available for KV cache ≈ device_RAM × 0.3 − model_weights_RAM

KV cache bytes = 2 × N × n_kv_heads × d_head × n_layers × 2  (BF16)

For a Q4_K_M 7B model on a 6 GB device:
  model weights ≈ 4.1 GB
  available for KV ≈ 6.0 × 0.3 − 4.1 ≈ negative → use Q2_K instead

For a Q2_K 7B model (≈2.7 GB) on a 6 GB device:
  available ≈ 6.0 × 0.3 − 2.7 ≈ -0.9 GB → still tight, use 2048 context max
  
For a Q4_K_M 3B model (≈1.8 GB) on a 6 GB device:
  available ≈ 6.0 × 0.45 − 1.8 ≈ 0.9 GB → 4096 context is feasible
```

A safe starting point: `contextSize = 2048` for 7B models, `contextSize = 4096` for 3B and smaller.

#### Production App Reference

For a production-ready implementation with system prompts, model management, benchmarks, and an Arm feature visualiser showing which CPU extensions are active on the current device:

```
Arm AI Chat — Google Play:
https://play.google.com/store/apps/details?id=com.arm.aichat
```

The app is open-reference for the full capability of the Android binding. The home screen shows which Arm features (NEON, dotprod, i8mm, SVE2, SME2) are present and active on the running device — useful for understanding the acceleration tier you are actually getting.

---

### M.3 Path 2 — Termux CLI (On-Device Native Build)

Termux provides a full Linux-like environment on Android without root. It is the fastest path from "I want to try this" to a running model.

#### Install Termux

```
Option A (recommended): F-Droid
  https://f-droid.org/en/packages/com.termux/

Option B: GitHub releases
  https://github.com/termux/termux-app/releases

Note: The Play Store version has reduced functionality due to Google
policy restrictions on executing downloaded code.
```

#### Environment Setup

```bash
# Update package index and upgrade installed packages
apt update && apt upgrade -y

# Install build tools
apt install git cmake clang

# Optional but recommended
apt install python3 python3-pip  # for model download scripts
```

#### Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Configure — Termux's clang supports ARM extensions natively
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON          # detect and use all CPU features present on this device

# Build (use all cores; expect 5–15 min on flagship)
cmake --build build --config Release -j$(nproc)
```

`GGML_NATIVE=ON` tells the compiler to use `-march=native`, which enables every extension the CPU reports: dotprod, i8mm, SVE2, and SME2 if present. This is safe because the binary only runs on this device — you are not cross-compiling for a different target.

#### Download a Model

```bash
# Create a models directory in home (fastest storage path on most devices)
mkdir -p ~/models

# Download from Hugging Face (example: Llama-3.2-3B Q4_K_M)
curl -L \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  -o ~/models/llama3.2-3b-q4km.gguf

# Alternative: use huggingface-cli if you have Python installed
pip install huggingface_hub
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --local-dir ~/models/
```

#### Run Inference

```bash
cd ~/llama.cpp

# Interactive chat
./build/bin/llama-cli \
  -m ~/models/llama3.2-3b-q4km.gguf \
  -c 4096 \
  --chat-template llama3 \
  -n 512 \
  --temp 0.7

# Single-shot prompt (good for scripting)
./build/bin/llama-cli \
  -m ~/models/llama3.2-3b-q4km.gguf \
  -c 2048 \
  -p "What is the capital of France?" \
  -n 128 \
  --no-display-prompt

# Run as local HTTP server (accessible from localhost on device)
./build/bin/llama-server \
  -m ~/models/llama3.2-3b-q4km.gguf \
  -c 4096 \
  --port 8080 \
  --host 127.0.0.1
```

**Critical: always set `-c` (context size)**. Without it, llama.cpp uses the model's maximum context (often 128K), which will immediately OOM-kill Termux on most phones. Start at 2048 and increase only if you have confirmed available RAM.

#### Performance Tips for Termux

```bash
# Check available RAM before loading
free -h

# Check which CPU extensions are active (shown at llama.cpp startup)
./build/bin/llama-cli --version   # prints build flags

# Use physical cores only — Android scheduler puts efficiency cores to sleep
# Snapdragon 8 Gen 3: 1 Prime + 3 Performance + 4 Efficiency
# Set threads to Prime + Performance count only:
-t 4   # for Snapdragon 8 Gen 3 (skip the 4 efficiency cores)

# Acquire a CPU wake lock to prevent throttling during long runs
# In a separate Termux session:
termux-wake-lock
```

---

### M.4 Path 3 — NDK Cross-Compilation (Host to Device)

Cross-compilation builds on your development machine and deploys to a connected Android device via ADB. This is the standard path for CI/CD pipelines and embedded products.

#### Prerequisites on Host

```
Android SDK with NDK r27+ installed
  Default path: ~/Library/Android/sdk/ndk/{version}/  (macOS)
                ~/Android/Sdk/ndk/{version}/           (Linux)
                %LOCALAPPDATA%\Android\sdk\ndk\{version}\  (Windows)

ADB installed and device connected with USB debugging enabled
```

#### Configure

```bash
export ANDROID_NDK=~/Library/Android/sdk/ndk/27.0.12077973  # adjust version

cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=armv8.7a" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -B build-android
```

**Flag rationale:**

| Flag | Reason |
|---|---|
| `ANDROID_ABI=arm64-v8a` | 64-bit ARM; covers virtually all devices since 2015 |
| `ANDROID_PLATFORM=android-28` | Android 9 minimum; required for full POSIX support |
| `-march=armv8.7a` | Enables i8mm and bf16 instructions present on most 2022+ flagship SoCs |
| `GGML_OPENMP=OFF` | NDK ships OpenMP but CMake integration is unreliable; use `-t` instead |
| `GGML_LLAMAFILE=OFF` | llamafile's fat binary approach does not support Android ABI |

For older or budget devices, drop to `-march=armv8.2a` for maximum compatibility. For cutting-edge devices with SME2 (Cortex-X925), use `-march=armv9.2a+sme2`, but note this narrows the device compatibility window significantly.

#### Build and Deploy

```bash
# Build (replace 8 with your core count)
cmake --build build-android --config Release -j8

# Install to a staging directory
cmake --install build-android \
  --prefix ./android-install \
  --config Release

# Push to device
adb shell "mkdir -p /data/local/tmp/llama"
adb push android-install/ /data/local/tmp/llama/

# Push your model
adb push ~/models/llama3.2-3b-q4km.gguf /data/local/tmp/llama/
```

#### Run via ADB Shell

```bash
adb shell
# Now inside the device shell:
cd /data/local/tmp/llama

# LD_LIBRARY_PATH is required — Android does not search relative lib/
LD_LIBRARY_PATH=lib ./bin/llama-cli \
  -m llama3.2-3b-q4km.gguf \
  -c 2048 \
  -p "Hello, I am running on Android." \
  -n 128
```

#### ABI Targeting by Device Class

```
Production release builds should target multiple ABIs:

  arm64-v8a   — all modern Android phones and tablets
  x86_64      — Android emulators, ChromeOS, Intel-based Android devices

For x86_64, replace the march flag:
  -DCMAKE_C_FLAGS="-march=x86-64-v3"
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3"

AMX (Advanced Matrix Extensions) on Intel x86-64 Android is auto-detected
at runtime by llama.cpp's CPUID path — no extra flags needed.
```

---

### M.5 Quantization Recommendations by Device Class

```
Device class   RAM    Recommended quant   Max context   Notes
─────────────────────────────────────────────────────────────────────
Budget phone   3–4 GB  Q2_K (3B model)    2048          7B will OOM
Mid-range      6 GB    Q4_K_M (3B) or     2048–4096     Test before shipping
                       Q2_K (7B)
Flagship       8–12 GB Q4_K_M (7B)        4096–8192     i8mm acceleration
               12 GB+  Q5_K_M (7B) or     8192          SVE2/SME2 if present
                       Q4_K_M (13B)
ChromeOS       16 GB+  Q4_K_M (13B)       8192+         x86-64, AMX
```

**Quantization format guide:**

| Format | Bits/weight (approx) | 7B model size | Quality loss |
|---|---|---|---|
| Q2_K | 2.6 | ~2.7 GB | Noticeable on reasoning tasks |
| Q3_K_M | 3.3 | ~3.4 GB | Moderate |
| Q4_K_M | 4.8 | ~4.1 GB | Minimal for most tasks |
| Q5_K_M | 5.6 | ~4.8 GB | Near-lossless |
| Q8_0 | 8.0 | ~7.2 GB | Essentially lossless |

For Android, Q4_K_M is the sweet spot for devices with ≥ 8 GB RAM. Q2_K or Q3_K_M for 6 GB devices, where you must leave 2–3 GB free for the OS and other apps.

---

### M.6 Memory and Thermal Management

```python
# Android OS memory pressure levels:
# TRIM_MEMORY_RUNNING_CRITICAL → free non-essential memory now
# TRIM_MEMORY_UI_HIDDEN        → app backgrounded, free aggressively
# onLowMemory()                → system-wide OOM imminent

# In your app's Activity/Service:
override fun onTrimMemory(level: Int) {
    if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
        engine.release()    // free the native context immediately
    }
}
```

Thermal throttling on Android is aggressive. A sustained inference session at full CPU speed will trigger thermal mitigation within 2–5 minutes on most devices, reducing clock speed by 30–50%. Mitigations:

```
1. Reduce thread count: -t 2 instead of -t 4 runs cooler at ~70% speed
2. Add inter-token delays in your app for conversational use cases
   (users tolerate 50 tokens/sec just as well as 80 tokens/sec)
3. Use flash attention: -fa flag reduces memory bandwidth and heat
4. Monitor: check /sys/class/thermal/thermal_zone*/temp via adb
```

---

## Part 2 — Apple Silicon

### M.7 Platform Overview

Apple Silicon (M1 through M4) has a fundamentally different memory architecture from both Android and data-center GPUs:

```
Apple Silicon unified memory architecture:
  ┌───────────────────────────────────────────────────────┐
  │                 Unified Memory Pool                    │
  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
  │  │ CPU (P+E)   │  │  GPU        │  │Neural Engine │ │
  │  │ 4–16 cores  │  │  10–40 core │  │  16–38 TOPS  │ │
  │  └─────────────┘  └─────────────┘  └──────────────┘ │
  │                                                        │
  │  RAM: 8–192 GB shared between ALL compute units       │
  │  Bandwidth: 100–800 GB/s (M1 base → M4 Max/Ultra)    │
  └───────────────────────────────────────────────────────┘

Key insight: there is no "VRAM limit" separate from system RAM.
A 70B model at Q4_K_M (≈38 GB) runs entirely on GPU on an M2 Ultra
with 64 GB, something impossible on a GPU with separate VRAM.
```

Acceleration stacks available to llama.cpp:

| Stack | Status | Activated by |
|---|---|---|
| CPU NEON/AMX | Always on | Default build |
| Metal (GPU) | Production-ready | `-DGGML_METAL=ON` |
| Core ML | Experimental | `-DGGML_COREML=ON` |
| ANE (Neural Engine) | Via Core ML only | Indirect |

**Metal is the primary acceleration path.** It offloads matrix multiplications to the GPU, which has higher memory bandwidth than the CPU (even though they share the same DRAM). For models that fit entirely in RAM, Metal provides 2–5× speedup over CPU-only inference depending on model size and quantization.

---

### M.8 Installation

#### Option A — Homebrew (Fastest Path)

```bash
brew install llama.cpp
```

The Homebrew formula builds with Metal enabled by default and is kept up to date by the maintainers. This is the right choice for most users who want to run models without building from source.

```bash
# Verify Metal is active
llama-cli --version
# Should show: ggml_metal_init: GPU name: Apple M[x]
```

#### Option B — Build from Source with CMake

```bash
# Prerequisites
xcode-select --install          # installs clang and Metal SDK
brew install cmake              # or use the Xcode-bundled cmake

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Configure with Metal (GPU) acceleration
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_NATIVE=ON              # use all CPU extensions (AMX on Apple Silicon)

cmake --build build --config Release -j$(sysctl -n hw.physicalcpu)
```

**Flag reference:**

| Flag | Effect |
|---|---|
| `GGML_METAL=ON` | Compile Metal shaders for GPU offload |
| `GGML_NATIVE=ON` | `-march=native`; enables Apple AMX instructions |
| `GGML_BLAS=ON` | Use Accelerate framework (BLAS); less important when Metal is on |
| `GGML_COREML=ON` | Experimental Core ML path; compiles `.mlpackage` at first load |

For a pure CPU build (useful for debugging or when GPU offload is undesirable):

```bash
cmake -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF \
  -DGGML_NATIVE=ON
```

---

### M.9 GPU Layer Offloading

The key runtime flag is `-ngl` (`--n-gpu-layers`). It controls how many transformer layers run on Metal vs. CPU.

```bash
# Fully CPU inference (no GPU)
./build/bin/llama-cli -m model.gguf -ngl 0

# Fully GPU inference (all layers on Metal)
./build/bin/llama-cli -m model.gguf -ngl 999   # 999 = offload everything

# Split: first 28 layers GPU, rest CPU
./build/bin/llama-cli -m model.gguf -ngl 28
```

**When to use split inference:**

If the model is larger than your unified memory allows at the desired context size, splitting layers between GPU and CPU is often better than reducing context:

```
GPU layers:  fast execution, uses high-bandwidth path
CPU layers:  slower, but the CPU AMX path is still competitive for large models

Rule of thumb for split:
  ngl = total_layers × (GPU_memory_budget / total_model_size)
  
  Example: 70B Q4_K_M (≈38 GB) on M1 Pro 16 GB:
    total_layers = 80
    You cannot fit the full model — use a smaller quant or larger machine.
    
  Example: 13B Q4_K_M (≈7.4 GB) on M1 base 8 GB:
    weights ≈ 7.4 GB, KV cache at 4K context ≈ 1.0 GB → tight but feasible
    ngl = 40 (all layers GPU) — it fits, no split needed
```

---

### M.10 Running Models

#### Download a Model

```bash
# Using Homebrew llama.cpp
llama-cli --hf-repo bartowski/Llama-3.2-3B-Instruct-GGUF \
          --hf-file Llama-3.2-3B-Instruct-Q4_K_M.gguf \
          -p "Hello" -n 10   # this triggers the download on first run

# Or explicitly with huggingface-cli
pip install huggingface_hub
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --local-dir ~/models/
```

#### Interactive Chat

```bash
./build/bin/llama-cli \
  -m ~/models/llama3.2-3b-q4km.gguf \
  -ngl 999 \
  -c 8192 \
  --chat-template llama3 \
  --temp 0.7 \
  -n -1               # generate until user stops
```

#### Run the HTTP Server

```bash
./build/bin/llama-server \
  -m ~/models/llama3.2-3b-q4km.gguf \
  -ngl 999 \
  -c 8192 \
  --port 8080 \
  --host 127.0.0.1 \
  --parallel 4 \        # concurrent requests (limited by context)
  --cont-batching       # continuous batching (Chapter 7.5)
```

This exposes an OpenAI-compatible API at `http://localhost:8080/v1`. You can point any OpenAI SDK to it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "What is metal shading language?"}]
)
print(response.choices[0].message.content)
```

#### Run as a Background Service (launchd)

For a persistent local inference server that survives reboots:

```xml
<!-- ~/Library/LaunchAgents/com.llamacpp.server.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.llamacpp.server</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/llama-server</string>
    <string>-m</string>
    <string>/Users/you/models/llama3.2-3b-q4km.gguf</string>
    <string>-ngl</string>
    <string>999</string>
    <string>-c</string>
    <string>8192</string>
    <string>--port</string>
    <string>8080</string>
    <string>--host</string>
    <string>127.0.0.1</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/llamacpp.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/llamacpp.err</string>
</dict>
</plist>
```

```bash
# Install and start
launchctl load ~/Library/LaunchAgents/com.llamacpp.server.plist
launchctl start com.llamacpp.server

# Stop
launchctl stop com.llamacpp.server
launchctl unload ~/Library/LaunchAgents/com.llamacpp.server.plist

# Check logs
tail -f /tmp/llamacpp.log
```

---

### M.11 Quantization Recommendations by Chip Tier

The unified memory architecture changes the quantization decision. On Apple Silicon, you are almost never memory-bandwidth constrained the way a discrete GPU is — you are constrained only by the total RAM available. This means you can often run a higher-quality quantization than you could on a GPU with separate VRAM.

```
Chip          Unified RAM   Recommended for 7B   Recommended for 70B
──────────────────────────────────────────────────────────────────────
M1 base       8 GB          Q4_K_M (tight)        Not feasible
M1 base       16 GB         Q5_K_M or Q6_K        Not feasible
M1 Pro        16 GB         Q6_K                  Not feasible
M1 Pro        32 GB         Q8_0                  Q2_K (tight)
M1 Max        32 GB         Q8_0                  Q3_K_M
M1 Max        64 GB         Q8_0                  Q5_K_M
M1 Ultra      64 GB         Q8_0                  Q5_K_M
M1 Ultra      128 GB        Q8_0                  Q8_0 (full quality)
M2 → M4       Same tiers    Same guidance          Same guidance
M4 Max        64 GB         Q8_0                  Q6_K
M4 Ultra      192 GB        Q8_0                  Q8_0 (+ 405B Q4_K_M)
```

Unlike discrete GPU inference where Q4_K_M is the practical maximum (VRAM limits), Apple Silicon lets you run Q8_0 — essentially lossless quantization — on all but the smallest memory configurations. This is the most compelling argument for running production workloads on Apple Silicon where absolute output quality matters.

---

### M.12 Performance Benchmarking

```bash
# Built-in benchmark tool
./build/bin/llama-bench \
  -m ~/models/llama3.2-3b-q4km.gguf \
  -ngl 999 \
  -p 512 \              # prompt tokens (prefill benchmark)
  -n 128 \              # generation tokens (decode benchmark)
  -r 3                  # repetitions for stable measurement

# Output format:
# model    | size  | params | backend | ngl | test   | t/s
# llama 3.2| 1.79G | 3.21B  | Metal   | 99  | pp 512 | 412.3 ± 2.1
# llama 3.2| 1.79G | 3.21B  | Metal   | 99  | tg 128 |  58.7 ± 0.4

# Reference numbers (approximate, vary by model and system load):
# M1 base  16 GB: 7B Q4_K_M tg ≈ 25–35 tok/s
# M2 Pro   16 GB: 7B Q4_K_M tg ≈ 45–55 tok/s
# M3 Max   48 GB: 7B Q4_K_M tg ≈ 80–100 tok/s
# M4 Max   64 GB: 7B Q4_K_M tg ≈ 120–150 tok/s
# M4 Ultra 192GB: 70B Q8_0  tg ≈ 35–45 tok/s
```

---

### M.13 Multi-Model and Production Configuration

For production deployments on macOS (e.g., a developer workstation serving a team):

```bash
# llama-server with multiple model slots
./build/bin/llama-server \
  -m ~/models/llama3.2-3b-q4km.gguf \
  --alias fast-model \
  -ngl 999 \
  -c 32768 \
  --port 8080 \
  --parallel 8 \
  --cont-batching \
  --flash-attn \                  # Flash Attention (Chapter 5) — significant speedup
  --mlock \                       # lock model weights in RAM, prevent paging
  --log-disable                   # reduce log verbosity in production
```

**Key production flags:**

| Flag | Effect |
|---|---|
| `--flash-attn` / `-fa` | Use Flash Attention kernel; cuts memory use and increases speed |
| `--mlock` | `mlock()` the model weights; prevents macOS from paging them out under pressure |
| `--parallel N` | Serve N concurrent requests via continuous batching |
| `--cont-batching` | Enable continuous batching (Chapter 7.5) |
| `--cache-type-k q8_0` | Quantize the KV cache keys to INT8; halves KV cache memory |
| `--cache-type-v q8_0` | Quantize the KV cache values to INT8 |
| `--threads-batch N` | Separate thread count for prefill (often set higher than decode threads) |

Quantizing the KV cache is particularly valuable on Apple Silicon because it stretches the context window without additional RAM:

```bash
# 70B Q4_K_M at 32K context with INT8 KV cache on M1 Ultra 128 GB
./build/bin/llama-server \
  -m ~/models/llama3-70b-q4km.gguf \
  -ngl 999 \
  -c 32768 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn \
  --mlock \
  --parallel 2 \
  --cont-batching
```

---

### M.14 Quick-Reference Comparison

| Dimension | Android (flagship) | Android (budget) | M1 base | M4 Max |
|---|---|---|---|---|
| RAM | 8–16 GB | 3–6 GB | 8–16 GB | 36–128 GB |
| GPU acceleration | Vulkan (limited) | None | Metal | Metal |
| Recommended max model | 7B Q4_K_M | 3B Q4_K_M | 13B Q4_K_M | 70B Q8_0 |
| Decode speed (7B Q4) | 20–40 tok/s | 5–15 tok/s | 25–35 tok/s | 100–150 tok/s |
| Thermal limit | 2–5 min at full speed | 1–2 min | Sustained (fan) | Sustained (fan) |
| Best use case | Edge / offline apps | Low-power IoT | Dev workstation | On-prem small team |
| Deployment path | ADB / Play Store | Termux / ADB | Homebrew / source | Homebrew / source |

---

*See Appendix B for installation of vLLM and llama.cpp on Linux/CUDA systems, Appendix D for the complete llama.cpp CLI flag reference, and Chapter 28 for llama.cpp as a general-purpose inference platform.*
