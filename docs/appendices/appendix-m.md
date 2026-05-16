# Appendix M — Mobile and Edge Deployment: Android and Apple Silicon

> *"The same model weights, the same GGUF format, the same attention mechanism — running on a phone in your pocket or a laptop on your desk. The inference engine is the only thing that changes."*

---

## Why This Appendix Exists

Chapters 1–13 assume a data-center context: CUDA GPUs, HBM memory, Linux servers, dedicated power supplies. But llama.cpp was explicitly designed to run anywhere a C++ compiler can reach — and that turns out to be almost everywhere. This appendix covers two of the most important non-data-center targets: **Android** (ARM and x86-64 mobile silicon) and **Apple Silicon** (the M-series unified-memory architecture).

Both platforms present constraints that require a genuinely different mental model from server deployment. On a server, you worry about utilisation and throughput. On a phone, you worry about battery drain, thermal throttling, and OS-level memory limits that can silently kill your process. On an Apple Silicon laptop, you discover that the absence of a PCIe bus between CPU and GPU changes the economics of model quantization in ways that no data-center textbook prepares you for.

By the end of this appendix you will understand three distinct paths to running llama.cpp on Android, how Apple Silicon's unified memory architecture changes the quantization calculus, how to configure Metal GPU acceleration, and how to run a persistent `llama-server` on macOS as a background service that survives reboots.

---

## Part 1 — Android

### M.1 Three Paths to Inference on Android

Android does not have a single way to run native C++ inference — it has three, each suited to a different use case. Understanding the distinction saves hours of confusion.

**Path 1: Android Studio GUI binding.** llama.cpp ships a complete Android application example in `examples/llama.android`. This is a proper Android app written in Kotlin that calls the llama.cpp C++ library through the **Java Native Interface (JNI)** — a bridge that lets Java/Kotlin code call functions written in C or C++. When a user taps "Run" in the app, Kotlin code calls a JNI function, which dispatches into the compiled llama.cpp library, which runs inference on the device hardware and returns tokens back through the same bridge. This path produces an `.apk` file you can ship through the Play Store. Hardware acceleration (SME2 on Arm, AMX on x86-64) is detected and enabled automatically at runtime.

**Path 2: Termux CLI.** Termux is a free Android app that gives you a full Linux-style terminal on your Android device without rooting it. Inside Termux you can install compilers, run `cmake`, and build llama.cpp from source directly on the device — then use `llama-cli` or `llama-server` exactly as you would on a Linux server. This path requires no Android development knowledge and is ideal for experimenting, running benchmarks, or building a personal device-side assistant. The main limitation is that Termux's compiler cannot target the GPU — you get CPU inference only.

**Path 3: NDK cross-compilation.** The Android NDK (Native Development Kit) is a toolchain that runs on your development machine (Linux or macOS) and produces binaries for Android. You build `llama-server` on your laptop and deploy the resulting binary to the phone using `adb` (Android Debug Bridge). This gives you fine-grained control over compiler flags, ABI targeting, and optimisation levels — and it is the right approach for CI/CD pipelines that need to produce Android binaries automatically.

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

### M.1.1 Hardware Acceleration on Android — SME2 and AMX Explained

Android devices use two main CPU instruction set extensions for accelerating matrix operations — the core computation in every transformer layer.

**SME2 (Scalable Matrix Extension 2)** is an Arm architecture extension introduced on high-end mobile SoCs (like recent Snapdragon chips). It adds dedicated matrix-multiply instructions that operate on large two-dimensional register tiles, dramatically accelerating the GEMM (General Matrix Multiply) operations that dominate LLM decode. llama.cpp detects SME2 at startup and selects the optimised kernel path automatically — you do not need to configure anything. The speedup over the scalar baseline can be 2–4× on chips that support it.

**AMX (Advanced Matrix Extensions)** is Intel's equivalent extension for x86-64 processors. Android devices running on x86-64 silicon (uncommon but used in some ChromeOS devices) benefit from AMX in the same way. Again, llama.cpp auto-detects it.

| CPU feature | Devices | Speedup vs baseline |
|---|---|---|
| SME2 | High-end Snapdragon (2024+) | 2–4× |
| NEON | All modern Arm Android devices | 1.5–2× |
| AMX | x86-64 Android / ChromeOS | 2–3× |
| Baseline (scalar) | Any | 1× |

The practical consequence: if you are benchmarking on an older device and the numbers look disappointing, check whether the device supports SME2. A Snapdragon 8 Gen 3 device will be materially faster than an 8 Gen 1 device running the same model and quantization, even though both run Android.

---

### M.2 Path 1: Android Studio GUI Application

This path is for engineers who want to ship a polished Android app. The `examples/llama.android` directory in the llama.cpp repository is a complete, working Android Studio project — not a skeleton, but a real app that loads a GGUF model, runs inference, and displays output in a chat-style UI.

#### M.2.1 Project Structure

```
examples/llama.android/
├── app/
│   ├── src/main/
│   │   ├── cpp/
│   │   │   └── CMakeLists.txt      # tells Gradle how to compile C++ code
│   │   ├── java/com/example/llama/
│   │   │   ├── MainActivity.kt     # UI logic (Kotlin)
│   │   │   └── Llm.kt              # JNI wrapper class (Kotlin ↔ C++ bridge)
│   │   └── AndroidManifest.xml
│   └── build.gradle
└── build.gradle
```

The JNI bridge in `Llm.kt` declares `external fun` functions — Kotlin's way of saying "this function is implemented in C++, not Kotlin." When the Kotlin code calls `generate(prompt)`, the Android runtime looks up the corresponding C++ function, transfers the arguments across the language boundary, and returns the result.

#### M.2.2 Build Steps

```bash
# On your development machine (Linux or macOS)

# 1. Install Android Studio from https://developer.android.com/studio
# 2. Open the project
cd llama.cpp/examples/llama.android
# Open this directory in Android Studio (File → Open)

# 3. Android Studio will prompt you to install the NDK.
#    Accept and let it install. This is the C++ toolchain.

# 4. Connect your Android device with USB debugging enabled:
#    Settings → Developer Options → USB Debugging

# 5. In Android Studio, press Run (▶) or:
./gradlew assembleDebug

# 6. Install the APK to connected device
adb install app/build/outputs/apk/debug/app-debug.apk
```

#### M.2.3 Loading a Model into the App

The app reads GGUF files from the device's storage. Copy a quantized model using `adb`:

```bash
# Push a model to the device's Downloads folder
adb push Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    /sdcard/Download/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Verify it landed correctly
adb shell ls -lh /sdcard/Download/*.gguf
```

Inside the app, use the file picker to navigate to Downloads and select the `.gguf` file. The app will begin loading — watch the logcat output in Android Studio for progress messages:

```bash
# Monitor device logs during model load
adb logcat -s llama.android:V
```

#### M.2.4 JNI Bridge — How Kotlin Talks to C++

For newcomers, the JNI pattern is worth understanding explicitly because it appears in every Android native code project.

In `Llm.kt`:
```kotlin
// The 'external' keyword marks this as a C++ function
external fun loadModel(path: String, nCtx: Int, nBatch: Int): Long
external fun generate(modelHandle: Long, prompt: String, nLen: Int): String
external fun freeModel(modelHandle: Long)

companion object {
    // This loads the compiled llama.cpp shared library at app startup
    init { System.loadLibrary("llama-android") }
}
```

In the C++ side (`llama_android.cpp`), the function names follow a specific convention — `Java_com_example_llama_Llm_loadModel` — that the JVM uses to find the corresponding native function. This convention is mechanical and the NDK tooling handles it automatically.

The `Long` returned by `loadModel` is a **handle** — a 64-bit integer that represents a pointer to the loaded model in C++ memory. Kotlin holds this integer and passes it back to C++ on subsequent calls, allowing the C++ code to locate the model without Kotlin needing to know anything about C++ memory management.

---

### M.3 Path 2: Termux CLI

Termux is the fastest way to get llama.cpp running on an Android device, particularly for development and experimentation. It requires no Android development knowledge and no host computer — everything happens on the device.

**What Termux actually is:** Termux installs a minimal Debian-like Linux userspace inside a sandboxed directory on your Android device. It is not a virtual machine and does not require root. It uses Android's native process model but provides Linux binaries, a package manager (`pkg`), and a shell. From inside Termux, your Android device looks like a small ARM Linux server.

```bash
# Install Termux from F-Droid (not from the Play Store — the Play Store
# version is outdated and the compiler packages may be missing)
# https://f-droid.org/packages/com.termux/

# Inside Termux — install build tools
pkg update && pkg upgrade -y
pkg install -y cmake git clang ninja python

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with the Termux compiler (this takes 10–20 minutes on most phones)
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -GNinja
cmake --build build --config Release -j$(nproc)
```

`-DGGML_NATIVE=ON` tells the compiler to generate instructions optimised for the exact CPU running the build — on a Snapdragon 8 Gen 3, this enables SME2 instructions if the compiler supports them.

```bash
# Run the chat interface
./build/bin/llama-cli \
    --model ~/storage/downloads/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --ctx-size 2048 \
    --n-predict 256 \
    -i -ins

# Or start a local HTTP server (accessible from other apps on the same device)
./build/bin/llama-server \
    --model ~/storage/downloads/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --ctx-size 2048 \
    --host 127.0.0.1 \
    --port 8080
```

**Accessing device storage from Termux:** By default Termux cannot read `/sdcard`. Grant storage access with:

```bash
termux-setup-storage
# Then access your files at ~/storage/downloads/
```

---

### M.4 Path 3: NDK Cross-Compilation

Cross-compilation means building on one machine (your development laptop) for a different target machine (an Android device). The result is a binary that runs on Android but was compiled on Linux or macOS. This is the right approach when you want to automate binary production in a CI pipeline, or when you need precise control over compilation flags that Termux's toolchain may not expose.

**What the NDK is:** The Android NDK (Native Development Kit) is a set of tools and headers that let you compile C and C++ code for Android. It includes a version of clang targeting Android's ABI, Android-specific system headers, and prebuilt libraries. You install it on your host machine, point cmake at it, and compile as if you were cross-compiling for an embedded Linux target.

```bash
# Install the NDK on your host (Linux or macOS)
# Option A: through Android Studio
#   SDK Manager → SDK Tools → NDK (Side by side) → Install

# Option B: command line (requires sdkmanager)
sdkmanager "ndk;27.0.12077973"

# Set the NDK path
export NDK_PATH="$HOME/Library/Android/sdk/ndk/27.0.12077973"
# On Linux: $HOME/Android/Sdk/ndk/27.0.12077973

# Clone llama.cpp on your host machine
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build for arm64-v8a (most modern Android phones)
cmake -B build-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DGGML_NATIVE=OFF   # OFF because host CPU ≠ target CPU
cmake --build build-android --config Release -j$(nproc)

# Deploy to connected device via adb
adb shell mkdir -p /data/local/tmp/llama
adb push build-android/bin/llama-server /data/local/tmp/llama/
adb push build-android/bin/llama-cli    /data/local/tmp/llama/

# Push the model
adb push Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    /data/local/tmp/llama/

# Run on-device
adb shell "chmod +x /data/local/tmp/llama/llama-server"
adb shell "/data/local/tmp/llama/llama-server \
    --model /data/local/tmp/llama/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --ctx-size 2048 --host 127.0.0.1 --port 8080 &"

# Forward device port to your laptop so you can query the server
adb forward tcp:8080 tcp:8080
curl http://localhost:8080/health
```

**ABI selection:** Android devices run on different CPU architectures. The `ANDROID_ABI` flag must match the device:

| ABI | Architecture | Device coverage |
|---|---|---|
| `arm64-v8a` | 64-bit Arm | All modern Android phones (2015+) |
| `x86_64` | 64-bit Intel/AMD | Emulators, ChromeOS, some tablets |
| `armeabi-v7a` | 32-bit Arm | Very old devices (pre-2015) — avoid |

For a CI pipeline targeting production phones, build `arm64-v8a` only.

---

### M.5 Quantization by Device Class

Quantization is the process of representing model weights at lower numerical precision to reduce memory usage and increase inference speed. A 7B model at full 16-bit (FP16) precision requires approximately 14 GB — far more than any Android device has. Quantized to 4 bits (Q4_K_M), the same model shrinks to about 4.4 GB. The quality difference is small enough that for most applications the quantized version is indistinguishable to users.

On Android, the choice of quantization is primarily constrained by available RAM. The OS reserves memory for itself and other running apps, so the usable headroom is always less than the device's advertised RAM.

| Device RAM | Practical headroom | Recommended quant | Max model size |
|---|---|---|---|
| 4 GB | ~2.5 GB | Q4_K_M | 1.5B params |
| 6 GB | ~4 GB | Q4_K_M | 3B params |
| 8 GB | ~5.5 GB | Q4_K_M | 3–7B params |
| 12 GB | ~9 GB | Q4_K_M or Q5_K_M | 7B params |
| 16 GB+ | ~13 GB | Q5_K_M or Q6_K | 7B–13B params |

**Quantization format glossary for newcomers:**

`Q4_K_M` — 4-bit weights with "k-quant" mixed precision (some sensitive layers stored at higher precision) and medium quality. The de facto standard for constrained devices; best balance of quality and size.

`Q5_K_M` — 5-bit k-quant medium. Noticeably better quality than Q4_K_M at about 25% larger file size. Use when you have the RAM headroom.

`Q2_K` — 2-bit k-quant. The most aggressive compression available. Quality degrades visibly. Use only when RAM is extremely tight and some quality loss is acceptable.

`Q8_0` — 8-bit quantization. Close to full-precision quality. File size is roughly 1× the parameter count in bytes (a 7B model ≈ 7 GB). Use on 16 GB+ devices when quality is paramount.

---

### M.6 Android Memory and Thermal Management

**Android's memory management is aggressive.** The OS can terminate background processes, including your inference server, at any time when it needs memory for foreground apps. This is not a bug — it is a core part of Android's design. The implications for a persistent inference service are significant:

- Running `llama-server` in a Termux session means it is subject to Android's low-memory killer. A phone call, a camera launch, or a memory-intensive game can cause Android to terminate the server process with no warning.
- The only reliable way to keep a service alive on Android is to run it as a **foreground service** with a persistent notification, or to accept that it may be killed and implement reconnection logic in your client.
- For development and personal use (Termux path), keep the screen on and consider disabling battery optimisation for Termux in Android Settings.

**Thermal throttling on mobile** is more aggressive than on desktop hardware. Modern Snapdragon chips are designed for burst performance — they sustain high clock speeds for 10–30 seconds then throttle down to prevent overheating. This means:

- The first few tokens after loading a model may arrive quickly (burst performance).
- Sustained inference over many minutes will be slower than the initial burst suggests.
- A benchmark that runs for 30 seconds will report higher throughput than one that runs for 5 minutes.

To get a realistic throughput number, run inference continuously for at least 3 minutes and measure the average over the final minute.

```bash
# Sustained throughput benchmark — run for 3 minutes, discard first minute
./build/bin/llama-bench \
    --model ~/storage/downloads/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    -p 512 -n 256 -r 5   # 5 repetitions for stable average
```

```bash
# Monitor CPU frequency to detect throttling
# (requires Termux with the termux-api package)
while true; do
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
    sleep 2
done
```

If the frequency drops significantly after 30–60 seconds of inference, throttling is happening. Active cooling (a phone cooler accessory) can extend the burst window on high-end devices.

---

## Part 2 — Apple Silicon

### M.7 Unified Memory — Why Apple Silicon Changes Everything

To understand Apple Silicon inference, you must first understand what **unified memory** means and why it is fundamentally different from the memory architecture of a discrete GPU system.

In a conventional PC or server with a dedicated GPU, there are two separate memory pools: system RAM (attached to the CPU via the memory controller) and VRAM (attached to the GPU via a separate memory controller, connected to the rest of the system over PCIe). When the GPU needs data that lives in system RAM — such as model weights that were loaded by the CPU — those weights must be physically transferred over the PCIe bus. PCIe Gen 4 x16 has a bandwidth of about 32 GB/s. A large model load or a mid-inference KV cache write that crosses this boundary pays the PCIe tax.

**Apple Silicon has no PCIe bus.** The CPU, GPU, Neural Engine, and memory controller are all on the same die, connected to a single shared LPDDR5X memory pool. When llama.cpp loads a model and allocates the KV cache, that memory is accessible to both the CPU compute cores and the GPU Metal shaders at full memory bandwidth — no copying, no transfer overhead. The bandwidth on an M3 Max is 400 GB/s, and every byte of it is available to whichever processor needs it at any given moment.

The practical consequence: on Apple Silicon, you can allocate your entire system RAM to a model. A MacBook Pro M3 Max with 96 GB of unified memory can comfortably run a 70B model at Q4_K_M (≈ 41 GB) and still have 55 GB for the OS, KV cache, and other applications. On a PC with a 24 GB GPU, the same model will not fit in VRAM and must be partially offloaded to CPU RAM — crossing the PCIe boundary on every forward pass.

This is why "how many GPU layers to offload" (`-ngl`) means something different on Apple Silicon than on a PC: on a PC, offloading layers to the GPU means moving weights into a physically separate VRAM pool, which is fast but finite. On Apple Silicon, offloading to the GPU (Metal) simply means asking the GPU cores to execute the matrix multiplications — the weights themselves do not move.

---

### M.8 Installation on Apple Silicon

**What Metal is:** Metal is Apple's GPU programming API, analogous to CUDA on NVIDIA hardware. Just as llama.cpp uses CUDA kernels to run matrix operations on NVIDIA GPUs, it uses Metal shaders to run those same operations on Apple's GPU cores. The Metal backend is maintained as part of the main llama.cpp codebase and is stable and performant.

```bash
# Install the Xcode Command Line Tools (provides the C++ compiler and Metal SDK)
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install cmake
brew install cmake

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with Metal support
# On Apple Silicon, cmake automatically detects and enables Metal.
# No special flag is required — the Metal backend is the default on macOS Arm.
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON    # explicitly enable if you want to be certain

cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
```

**Verify Metal is active:**

```bash
./build/bin/llama-cli --version
# Should include: Metal

# Run a quick test — watch for Metal allocation messages
./build/bin/llama-cli \
    --model /path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 99 \
    -p "The capital of France is" \
    -n 20
# Look for: "llm_load_tensors: ggml ctx size = ... Metal"
```

---

### M.9 GPU Layer Offloading with `-ngl`

The `-ngl` flag (short for `--n-gpu-layers`) controls how many transformer layers are executed by the GPU rather than the CPU. Understanding it requires knowing what a "layer" means in the context of a transformer model.

A transformer model is a stack of identical blocks, each containing a self-attention sublayer and a feed-forward network sublayer. A 7B parameter Llama model has 32 such blocks. A 70B model has 80 blocks. The `--n-gpu-layers` flag tells llama.cpp to assign the last N of those blocks to the GPU — the rest run on the CPU.

On Apple Silicon, because of unified memory, you will almost always want to set `-ngl` high enough to offload all layers. The GPU cores on an M-series chip are substantially faster at matrix multiply than the CPU cores, and offloading to the GPU does not consume a separate memory pool — it just changes which compute units do the work.

```bash
# Fully GPU-accelerated — recommended for all Apple Silicon
llama-server \
    --model /path/to/model.gguf \
    --n-gpu-layers 999 \   # 999 = offload everything; excess is ignored
    --ctx-size 8192 \
    --host 0.0.0.0 \
    --port 8080

# Watch startup to confirm GPU allocation
# You should see lines like:
# llm_load_tensors: offloading 32 repeating layers to GPU
# llm_load_tensors: offloaded 32/32 layers to GPU
# ggml_metal_init: allocating
```

**When might you reduce `-ngl`?** Only if you are running the model alongside other Metal workloads (video rendering, CoreML inference) that need GPU memory, or if you are on a base M1 with 8 GB where you need to leave more unified memory for the OS. In that case, try `-ngl 24` (offloading the upper three-quarters of layers) as a starting point.

---

### M.10 Running a Model and Persistent Background Service

**Single-shot inference:**

```bash
# Basic chat interface
./build/bin/llama-cli \
    --model /Users/you/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 4096 \
    -i -ins   # interactive instruction-following mode
```

**HTTP server for API access:**

```bash
./build/bin/llama-server \
    --model /Users/you/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 4096 \
    --parallel 2 \
    --cont-batching \
    --host 0.0.0.0 \
    --port 8080
```

### M.10.1 Running llama-server as a Persistent macOS Service

**What launchd is:** On macOS, `launchd` is the system service manager — it is to macOS what `systemd` is to Linux. You describe a service in a `.plist` (property list) XML file, place it in `~/Library/LaunchAgents/`, and `launchd` starts it at login, restarts it if it crashes, and gives you `launchctl` commands to control it. This is how you turn `llama-server` from something you start manually in a terminal into a background service that is always available.

```xml
<!-- ~/Library/LaunchAgents/com.local.llama-server.plist -->
<!--
  This file tells macOS launchd to start llama-server at login
  and restart it automatically if it exits.
  
  Key fields:
  - Label: a unique identifier for this service
  - ProgramArguments: the command to run (first element is the binary,
    subsequent elements are arguments — NOT a shell command string)
  - RunAtLoad: true means start immediately when this plist is loaded
  - KeepAlive: true means restart if the process exits for any reason
  - StandardOutPath / StandardErrorPath: log files for debugging
-->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.llama-server</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/you/llama.cpp/build/bin/llama-server</string>
        <string>--model</string>
        <string>/Users/you/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf</string>
        <string>--n-gpu-layers</string>
        <string>999</string>
        <string>--ctx-size</string>
        <string>4096</string>
        <string>--parallel</string>
        <string>2</string>
        <string>--cont-batching</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8080</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/tmp/llama-server.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/llama-server.err</string>

    <!-- Give the process enough file descriptors for parallel connections -->
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>65536</integer>
    </dict>
</dict>
</plist>
```

```bash
# Load and start the service
launchctl load ~/Library/LaunchAgents/com.local.llama-server.plist

# Check that it's running
launchctl list | grep llama
curl http://127.0.0.1:8080/health

# View logs
tail -f /tmp/llama-server.log

# Stop and unload the service
launchctl unload ~/Library/LaunchAgents/com.local.llama-server.plist

# After editing the plist file, reload it with:
launchctl unload ~/Library/LaunchAgents/com.local.llama-server.plist
launchctl load   ~/Library/LaunchAgents/com.local.llama-server.plist
```

---

### M.11 Quantization by Apple Silicon Tier

Apple Silicon chips vary enormously in unified memory capacity. The base M1 with 8 GB is a different world from the M4 Ultra with 192 GB. Here is the practical quantization guide:

| Chip | Memory | Max practical model | Recommended quant | Notes |
|---|---|---|---|---|
| M1 / M2 (8 GB) | 8 GB | 7B | Q4_K_M | Leave 3 GB for OS + KV cache |
| M1 / M2 (16 GB) | 16 GB | 13B | Q4_K_M | Comfortable for 7B at Q8_0 |
| M1 / M2 Pro (16 GB) | 16 GB | 13B | Q4_K_M | Same memory, better CPU |
| M1 / M2 Pro (32 GB) | 32 GB | 13B | Q8_0 or 7B F16 | |
| M1 / M2 Max (64 GB) | 64 GB | 70B | Q4_K_M | Comfortable 34B at Q5_K_M |
| M1 / M2 Ultra (96–192 GB) | Up to 192 GB | 70B+ | Q8_0 or higher | Research-grade local inference |
| M3 / M4 (base, 16 GB) | 16 GB | 13B | Q4_K_M | Faster than M1/M2 base |
| M3 / M4 Max (48–128 GB) | Up to 128 GB | 70B | Q4_K_M to Q8_0 | Best perf/watt on laptop |
| M4 Ultra (192 GB) | 192 GB | 405B | Q4_K_M | Full Llama-3 405B fits |

**Memory budget formula:** `Available for inference = Total RAM − 2 GB (OS) − KV cache size`. The KV cache for a 7B model at 4K context in fp16 is roughly 2 GB; at 32K context it is roughly 16 GB. Reduce `--ctx-size` if you are pushing the memory limit.

---

### M.12 Benchmarking on Apple Silicon

```bash
# Standard throughput benchmark — run 3 times for stable average
./build/bin/llama-bench \
    --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    -p 512 \    # prompt tokens (tests prefill speed)
    -n 128 \    # generation tokens (tests decode speed — what users experience)
    -r 3        # repetitions

# Expected output format:
# model      | n_gpu_layers | test    | t/s
# Qwen2.5-7B | 99           | pp512   | 720.45 ± 3.2   (prefill)
# Qwen2.5-7B | 99           | tg128   | 43.21  ± 0.8   (decode)
```

**Reading the output:** `pp` is prompt processing (prefill) throughput — how fast the model processes your input prompt. `tg` is token generation (decode) throughput — how many tokens per second the model generates. Users experience the `tg` number: 43 tokens/sec is comfortable for interactive use; below 10 tokens/sec feels noticeably slow.

**Monitoring memory pressure during inference:**

```bash
# macOS memory pressure indicator (open in another terminal during inference)
vm_stat 2 | awk 'NR>1 { printf "Free: %.1f GB  Wired: %.1f GB\n",
    $1*4096/1e9, $6*4096/1e9 }'

# More detailed — Activity Monitor → Memory tab → Memory Pressure graph
# Green = healthy, Yellow = moderate pressure, Red = swap is being used heavily
```

If the memory pressure graph turns red during inference, the model is too large for your configuration. Reduce `--ctx-size` first, then consider a more aggressive quantization.

---

### M.13 Production Configuration for Apple Silicon

```bash
# Production-grade llama-server for an Apple Silicon Mac used as a local API endpoint
./build/bin/llama-server \
    --model /Users/you/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --n-gpu-layers 999 \      # all layers to Metal GPU
    --ctx-size 8192 \         # 8K context window
    --parallel 4 \            # 4 concurrent request slots
    --cont-batching \         # batch tokens from multiple requests together (Chapter 3)
    --flash-attn \            # Flash Attention: reduces memory BW for long contexts (Chapter 5)
    --mlock \                 # lock model weights in RAM; prevent OS from swapping them
    --batch-size 512 \        # prefill batch size
    --ubatch-size 128 \       # micro-batch size for decode
    --host 0.0.0.0 \
    --port 8080 \
    --metrics \               # expose /metrics endpoint for Prometheus scraping
    --log-prefix \
    --log-timestamps
```

**Flag explanations for newcomers:**

`--cont-batching` enables continuous batching — the scheduler described in Chapter 3. Rather than processing one request at a time, the server batches tokens from multiple in-flight requests together in each GPU pass. This dramatically improves throughput when multiple clients are active simultaneously, at no quality cost.

`--flash-attn` enables the Flash Attention algorithm (Chapter 5). Instead of materialising the full N×N attention score matrix in memory — which grows quadratically with context length — Flash Attention computes attention in tiles that fit in fast on-chip memory. On Apple Silicon this reduces the memory bandwidth cost of long contexts and can increase decode speed by 20–30% at 8K+ context lengths.

`--mlock` locks the model weights into RAM, preventing macOS from paging them to disk under memory pressure. Without this flag, an OS-level memory event (another app allocating a large buffer) can cause model weights to be swapped to disk mid-inference, producing a severe latency spike. On a system dedicated to inference, always use `--mlock`.

---

### M.14 Android vs Apple Silicon — Quick Comparison

| Dimension | Android (high-end) | Apple Silicon |
|---|---|---|
| **GPU acceleration** | SME2/NEON (CPU SIMD), no Metal | Metal (full GPU) |
| **Memory architecture** | Separate CPU/GPU pools or fully shared (SoC-dependent) | Fully unified — no copy cost |
| **Max practical model** | 7B Q4 on 12 GB device | 70B Q4 on 64 GB M2 Max |
| **Thermal throttling** | Aggressive (minutes) | Moderate (hours before throttle) |
| **Persistent service** | Foreground service or accept termination | launchd plist — stable |
| **Install complexity** | Three paths; toolchain setup required | Homebrew + cmake |
| **Best use case** | On-device private inference, mobile apps | Developer laptop, local API endpoint |
| **Battery impact** | High — limit inference duration | Manageable — M-series very efficient |

---

*For Raspberry Pi and NVIDIA Jetson deployment, see Appendix N. For the general llama.cpp CLI flag reference, see Appendix D. For quantization internals (GGUF, AWQ, FP8), see Chapter 10.*

---

## Part 3 — MLX: Apple's Native ML Framework for Apple Silicon

### M.15 What MLX Is

llama.cpp running on Metal is one way to use Apple Silicon for LLM inference.
Apple's own **MLX** framework is another — and for many Apple Silicon
deployment scenarios it is the better choice.

MLX (Machine Learning eXtended) is an open-source array framework developed
by Apple specifically for Apple Silicon. It is not a port of PyTorch or a
wrapper around CoreML. It is built from scratch to exploit the unified memory
architecture of M-series chips and the Metal shader compilation pipeline.

The two defining design choices:

**Unified memory as a first-class abstraction**: MLX arrays live in unified
memory. There is no explicit `to('cpu')` or `to('mps')` call — the same array
is accessible by CPU and GPU simultaneously. Operations transparently dispatch
to whichever compute unit is fastest.

**Lazy evaluation with graph compilation**: MLX operations are lazy by default.
When you write `y = x @ W`, nothing executes. MLX records the operation in a
computation graph and executes the entire graph when you call `mx.eval()`.
This enables kernel fusion: the compiler combines adjacent operations into
single Metal shaders, eliminating intermediate memory writes.

```python
import mlx.core as mx
import mlx.nn as nn

# All operations are lazy — no execution yet
x = mx.array([[1.0, 2.0, 3.0]])
W = mx.random.normal((3, 4))
y = x @ W                          # recorded but not computed
z = nn.gelu(y)                      # fused into same graph

mx.eval(z)                          # execute: single optimised Metal shader
```

### M.16 MLX Architecture: Memory and Compute

**Unified memory benefits for LLM inference:**

On NVIDIA GPUs, loading a 7B FP16 model requires copying ~14 GB from CPU RAM
to GPU VRAM over PCIe (bandwidth: 32 GB/s → ~0.44 seconds just for the copy).
On Apple Silicon, the model weights are in unified memory from the start. The
GPU reads them directly with the same HBM bandwidth the CPU uses.

```
Apple M3 Max unified memory bandwidth: 300 GB/s
NVIDIA RTX 4090 VRAM bandwidth: 1,008 GB/s (but PCIe bottleneck at 64 GB/s)
```

For inference-bound workloads (memory-bandwidth-limited GEMV during decode),
the effective bandwidth for a 7B model:

```
M3 Max (36GB):   300 GB/s effective (no copy)
RTX 3090 (24GB): 936 GB/s VRAM bandwidth, but model must fit in 24GB
RTX 4060 (8GB):  288 GB/s — similar effective bandwidth to M3 Max
```

For models that fit in Apple Silicon's unified memory, the performance is
comparable to a similarly-priced consumer NVIDIA GPU.

### M.17 mlx-lm: LLM Inference with MLX

`mlx-lm` is the HuggingFace-compatible LLM inference library built on MLX.
It supports the full Llama, Mistral, Gemma, Phi, and Qwen families.

```bash
pip install mlx-lm

# One-command inference
mlx_lm.generate \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --prompt "What is the capital of France?" \
    --max-tokens 100

# Start an OpenAI-compatible server
mlx_lm.server \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --port 8080
```

```python
from mlx_lm import load, generate

# Load model (downloads from HuggingFace Hub if needed)
model, tokenizer = load("mlx-community/Llama-3.1-8B-Instruct-4bit")

# Generate
response = generate(
    model, tokenizer,
    prompt="Explain attention in one paragraph.",
    max_tokens=200,
    verbose=True     # print tokens as they generate
)
print(response)
```

### M.18 MLX Quantization

MLX uses its own quantization format (`.npz` weight files with quantization
metadata), distinct from GGUF. The `mlx-lm` library handles conversion:

```bash
# Convert and quantize from HuggingFace
python -m mlx_lm.convert \
    --hf-path meta-llama/Llama-3.1-8B-Instruct \
    --mlx-path ./Llama-3.1-8B-4bit \
    --quantize \
    --q-bits 4         # INT4 (default)
    # --q-bits 8       # INT8 for higher quality

# Or download pre-converted models from mlx-community
# mlx-community has pre-quantized versions of most popular models
```

The MLX quantization uses a group size of 64 (vs llama.cpp's Q4_K_M which
uses group size 32). This is slightly less accurate than Q4_K_M but faster
due to vectorized group-scale application on Apple Silicon's SIMD units.

### M.19 Performance Comparison: mlx-lm vs llama.cpp Metal

| Model | Hardware | mlx-lm (INT4) | llama.cpp Q4_K_M | Notes |
|---|---|---|---|---|
| Llama 3.1 8B | M3 Max (36GB) | 62 tok/s | 55 tok/s | MLX +13% |
| Llama 3.1 70B | M2 Ultra (192GB) | 14 tok/s | 11 tok/s | MLX +27% |
| Phi-4 (14B) | M3 Max (36GB) | 38 tok/s | 33 tok/s | MLX +15% |
| Gemma 3 12B | M3 Max (36GB) | 41 tok/s | 37 tok/s | MLX +11% |

MLX is consistently faster than llama.cpp Metal on Apple Silicon for two
reasons: the Metal shaders are compiled specifically for Apple's GPU
microarchitecture (vs llama.cpp's more generic Metal kernels), and the lazy
evaluation enables fusion that eliminates intermediate buffers.

The gap narrows at larger batch sizes (mlx-lm does not yet match llama.cpp's
continuous batching for server workloads). For single-user inference, prefer
mlx-lm. For serving multiple concurrent users, llama.cpp with `--n-parallel`
is currently better-optimised.

### M.20 MLX Low-Level: Writing Custom Kernels

For advanced users who want to write custom Metal kernels within the MLX
framework:

```python
import mlx.core as mx

# Custom Metal kernel via mx.fast.metal_kernel
source = """
kernel void scaled_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = a[idx] + scale * b[idx];
}
"""

kernel = mx.fast.metal_kernel(
    name="scaled_add",
    input_names=["a", "b", "scale"],
    output_names=["out"],
    source=source
)

a = mx.ones((1024,))
b = mx.ones((1024,))
scale = mx.array(2.0)

out = kernel(inputs=[a, b, scale], output_shapes=[(1024,)],
             output_dtypes=[mx.float32], grid=(1024,1,1), threadgroup=(256,1,1))
mx.eval(out)
```

This is useful for custom quantization dequantization kernels, custom attention
variants (e.g. sliding window), or activation functions not in the MLX standard
library.

### M.21 When to Choose mlx-lm vs llama.cpp on Apple Silicon

| Scenario | Recommendation | Reason |
|---|---|---|
| Single user, maximum speed | mlx-lm | Faster Metal kernels |
| Multi-user server (>4 concurrent) | llama.cpp | Better batching/scheduling |
| Model not in mlx-community | llama.cpp | Wider GGUF format support |
| Custom quantization format | llama.cpp | GGUF ecosystem is larger |
| Embedding model serving | llama.cpp | mlx-lm embedding support limited |
| Production API server | llama.cpp | More mature server mode |
| Developer experimentation | mlx-lm | Simpler Python API |
| Custom Metal kernel work | mlx-lm | `mx.fast.metal_kernel` API |

### M.22 Model Availability

The `mlx-community` organisation on HuggingFace maintains pre-converted and
pre-quantized MLX versions of the most popular models:

```bash
# Browse available models
# https://huggingface.co/mlx-community

# Common models available as MLX:
mlx-community/Llama-3.1-8B-Instruct-4bit
mlx-community/Llama-3.1-70B-Instruct-4bit
mlx-community/Llama-3.3-70B-Instruct-4bit
mlx-community/Phi-4-4bit
mlx-community/gemma-3-27b-it-4bit
mlx-community/Mistral-7B-Instruct-v0.3-4bit
mlx-community/Qwen2.5-72B-Instruct-4bit
mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit
```

If a model is not yet available in MLX format, convert it with `mlx_lm.convert`
(§M.18) or use llama.cpp with the GGUF version.

---

*For Raspberry Pi and NVIDIA Jetson deployment, see Appendix N. For the general llama.cpp CLI flag reference, see Appendix D. For quantization internals (GGUF, AWQ, FP8), see Chapter 10. For MLX low-level Metal kernel programming, see the Metal shader concepts in Appendix J.*
