# Chapter 29: Multimodal Inference — Companion Code

## Python — `multimodal_demo.py`

```python
# multimodal_demo.py
"""
Chapter 29 — Multimodal Inference Demo (Python)

Simulates the full VLM pipeline:
  1. Image patch extraction and token count calculation
  2. Memory cost analysis across VLM families
  3. Tile strategy simulation (LLaVA-1.6 style)
  4. Batch throughput modeling for vision workloads
  5. Audio (Whisper) pipeline overview
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATCH EXTRACTION AND TOKEN COUNT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionEncoderSpec:
    name: str
    patch_size: int      # pixels per patch (one side)
    input_resolution: int  # native encoder input size (square)
    output_dim: int      # embedding dimension
    n_transformer_layers: int
    params_m: float      # million parameters


@dataclass
class VLMSpec:
    name: str
    encoder: VisionEncoderSpec
    projection: str       # "linear" | "mlp" | "pixel_shuffle" | "cross_attention"
    compression_ratio: float  # patch compression (1.0 = no compression)
    llm_layers: int
    llm_kv_heads: int
    llm_head_dim: int
    llm_dim: int
    max_tiles: int        # max sub-image tiles
    thumbnail: bool       # include global thumbnail
    max_resolution: int   # maximum input resolution (one side)


CLIP_VIT_L_336 = VisionEncoderSpec(
    name="CLIP ViT-L/14@336",
    patch_size=14, input_resolution=336,
    output_dim=1024, n_transformer_layers=24, params_m=307,
)

SIGLIP_SO400M = VisionEncoderSpec(
    name="SigLIP ViT-SO400M",
    patch_size=14, input_resolution=384,
    output_dim=1152, n_transformer_layers=27, params_m=400,
)

INTERN_VIT_300M = VisionEncoderSpec(
    name="InternViT-300M",
    patch_size=14, input_resolution=448,
    output_dim=1024, n_transformer_layers=24, params_m=300,
)

VLM_FAMILY = [
    VLMSpec("LLaVA-1.5-7B",   CLIP_VIT_L_336,  "mlp",           1.0, 32, 8, 128, 4096, 1,  False, 336),
    VLMSpec("LLaVA-1.6-7B",   CLIP_VIT_L_336,  "pixel_shuffle", 1.0, 32, 8, 128, 4096, 4,  True,  672),
    VLMSpec("InternVL2-8B",   INTERN_VIT_300M, "pixel_shuffle", 4.0, 32, 8, 128, 4096, 12, True,  4032),
    VLMSpec("Qwen2-VL-7B",    SIGLIP_SO400M,   "mlp",           1.0, 28, 4, 128, 3584, 1,  False, 2048),
    VLMSpec("MiniCPM-V-2.6",  SIGLIP_SO400M,   "pixel_shuffle", 9.0, 32, 8, 128, 4096, 9,  True,  1800),
]


def patches_per_tile(encoder: VisionEncoderSpec) -> int:
    """Number of patches from one encoder tile."""
    n = encoder.input_resolution // encoder.patch_size
    return n * n


def visual_tokens_for_image(
    vlm: VLMSpec, image_w: int, image_h: int
) -> Tuple[int, int, str]:
    """
    Returns (n_visual_tokens, n_tiles, description).
    """
    enc = vlm.encoder
    base_patches = patches_per_tile(enc)
    tokens_per_tile = int(base_patches / vlm.compression_ratio)

    if vlm.max_tiles == 1:
        # Fixed resolution
        n_tiles = 1
        desc = f"fixed {enc.input_resolution}×{enc.input_resolution}"
    else:
        # Dynamic tiling
        max_dim = max(image_w, image_h)
        effective_res = min(max_dim, vlm.max_resolution)
        # Number of tiles in each dimension
        tiles_w = max(1, round(image_w / enc.input_resolution))
        tiles_h = max(1, round(image_h / enc.input_resolution))
        # Cap to max_tiles
        n_content_tiles = min(tiles_w * tiles_h, vlm.max_tiles)
        n_tiles = n_content_tiles + (1 if vlm.thumbnail else 0)
        desc = f"{tiles_w}×{tiles_h} tiles + {'thumbnail' if vlm.thumbnail else 'no thumbnail'}"

    total_tokens = n_tiles * tokens_per_tile
    return total_tokens, n_tiles, desc


def kv_bytes_for_visual_tokens(vlm: VLMSpec, n_tokens: int, dtype_bytes: int = 2) -> int:
    return 2 * vlm.llm_layers * vlm.llm_kv_heads * vlm.llm_head_dim * n_tokens * dtype_bytes


def print_token_table():
    test_images = [
        (336, 336,   "Square thumbnail"),
        (672, 672,   "Standard HD"),
        (1024, 768,  "Landscape photo"),
        (768, 1024,  "Portrait photo"),
        (1920, 1080, "Full HD screenshot"),
        (2048, 2048, "High-res scan"),
    ]

    print("=" * 90)
    print("Visual Token Count and KV Cost by VLM and Image Resolution")
    print("=" * 90)

    for img_w, img_h, img_desc in test_images:
        print(f"\n  Image: {img_w}×{img_h}  ({img_desc})")
        print(f"  {'Model':22}  {'Tokens':8}  {'KV MB':8}  {'Tiling strategy'}")
        print("  " + "-" * 68)
        for vlm in VLM_FAMILY:
            n_tok, n_tiles, desc = visual_tokens_for_image(vlm, img_w, img_h)
            kv_mb = kv_bytes_for_visual_tokens(vlm, n_tok) / 1e6
            print(f"  {vlm.name:22}  {n_tok:8,}  {kv_mb:7.1f}M  {desc}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREFILL COST MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModalityPrefillCost:
    """Prefill cost breakdown for one multimodal request."""
    vision_encode_ms: float    # ViT forward pass
    projection_ms: float       # projection layer
    llm_prefill_ms: float      # LLM processing visual + text tokens
    total_ttft_ms: float       # sum


def estimate_prefill_cost(
    vlm: VLMSpec,
    n_visual_tokens: int,
    n_text_tokens: int,
    h100_tflops: float = 989.0,
) -> ModalityPrefillCost:
    """
    Rough estimate of prefill wall time on H100.
    Vision encoder: ~2 FLOPs/param per forward pass
    LLM: ~2 × param_count FLOPs per token
    """
    enc = vlm.encoder

    # Vision encoder FLOPs: ViT with n_transformer_layers attention blocks
    # Each block: ~4 * d² * n_patches (self-attention) + ~8 * d² * n_patches (FFN)
    n_patches = patches_per_tile(enc)
    vis_flops  = enc.n_transformer_layers * 12 * (enc.output_dim ** 2) * n_patches
    vis_ms     = vis_flops / (h100_tflops * 1e12) * 1e3

    # Projection: 2-layer MLP of dim d_vis → d_llm
    proj_flops = 2 * n_visual_tokens * enc.output_dim * vlm.llm_dim
    proj_ms    = proj_flops / (h100_tflops * 1e12) * 1e3

    # LLM prefill: 2 × n_params × n_tokens FLOPs (approx)
    # Llama-3.1-8B has ~8B params
    llm_params  = vlm.llm_layers * 4 * (vlm.llm_dim ** 2)  # rough (attention + FFN)
    total_tokens = n_visual_tokens + n_text_tokens
    llm_flops   = 2 * llm_params * total_tokens
    llm_ms      = llm_flops / (h100_tflops * 1e12) * 1e3

    total_ms = vis_ms + proj_ms + llm_ms
    return ModalityPrefillCost(vis_ms, proj_ms, llm_ms, total_ms)


def print_prefill_breakdown():
    vlm = VLM_FAMILY[1]  # LLaVA-1.6-7B
    image_configs = [
        (576,  512, "LLaVA-1.5 style (1 tile, 512 text)"),
        (2880, 512, "LLaVA-1.6 HD (4 tiles + thumb, 512 text)"),
        (2880, 128, "LLaVA-1.6 HD (4 tiles, short prompt)"),
        (576,  0,   "Image-only (no text, 576 visual tokens)"),
    ]

    print("=" * 75)
    print("Prefill Cost Breakdown (H100, LLaVA-1.6-7B BF16)")
    print("=" * 75)
    print(f"  {'Config':42}  {'ViT':8}  {'Proj':6}  {'LLM':8}  {'TTFT':8}")
    print("  " + "-" * 74)
    for n_vis, n_text, desc in image_configs:
        cost = estimate_prefill_cost(vlm, n_vis, n_text)
        print(f"  {desc:42}  {cost.vision_encode_ms:6.1f}ms  "
              f"{cost.projection_ms:4.1f}ms  "
              f"{cost.llm_prefill_ms:6.1f}ms  "
              f"{cost.total_ttft_ms:6.1f}ms")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TILE STRATEGY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def optimal_tiling(
    img_w: int, img_h: int,
    tile_size: int,
    max_tiles: int,
) -> Tuple[int, int, int]:
    """
    Find the tiling (tiles_w, tiles_h) that best preserves aspect ratio
    without exceeding max_tiles.
    Returns (tiles_w, tiles_h, total_tiles).
    """
    aspect = img_w / img_h
    best = (1, 1)
    best_score = float("inf")

    for tw in range(1, max_tiles + 1):
        for th in range(1, max_tiles + 1):
            if tw * th > max_tiles:
                continue
            ratio = tw / th
            score = abs(ratio - aspect)
            if score < best_score:
                best_score = score
                best = (tw, th)

    tw, th = best
    return tw, th, tw * th


def print_tiling_table():
    print("=" * 65)
    print("Optimal Tiling Strategy (LLaVA-1.6, max_tiles=4)")
    print("=" * 65)

    images = [
        (336, 336),
        (672, 336),
        (336, 672),
        (1024, 576),
        (576, 1024),
        (800, 600),
        (1920, 1080),
        (1280, 720),
    ]

    print(f"  {'Image':14}  {'Aspect':8}  {'Tiling':10}  {'Tiles':6}  "
          f"{'Tokens (w/ thumb)':20}")
    print("  " + "-" * 62)
    for w, h in images:
        tw, th, n_tiles = optimal_tiling(w, h, 336, 4)
        aspect = w / h
        with_thumb = (n_tiles + 1) * 576  # 576 per tile, +1 thumbnail
        print(f"  {w}×{h:4}        {aspect:8.2f}  {tw}×{th:<7}   "
              f"{n_tiles:3}    {with_thumb:,}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  AUDIO PIPELINE OVERVIEW (WHISPER)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WhisperSpec:
    name: str
    n_encoder_layers: int
    n_decoder_layers: int
    d_model: int
    n_heads: int
    params_m: float
    languages: int
    wer_librispeech: float  # word error rate on LibriSpeech clean


WHISPER_MODELS = [
    WhisperSpec("tiny",   4,  4,  384, 6,   39,   99, 5.7),
    WhisperSpec("base",   6,  6,  512, 8,   74,   99, 4.2),
    WhisperSpec("small",  12, 12, 768, 12,  244,  99, 3.0),
    WhisperSpec("medium", 24, 24, 1024, 16, 769,  99, 2.0),
    WhisperSpec("large",  32, 32, 1280, 20, 1550, 99, 2.7),
    WhisperSpec("large-v3", 32, 32, 1280, 20, 1550, 100, 2.0),
    WhisperSpec("turbo",  32, 2,  1280, 20, 809,  99, 2.7),
]


def whisper_audio_to_tokens(audio_seconds: float) -> int:
    """
    Whisper processes audio in 30-second chunks.
    Each 30s chunk → 3000 Mel spectrogram frames (25ms hop, 25ms window).
    These are processed by the audio encoder as a fixed sequence of 1500 tokens
    (after 2× temporal downsampling in the encoder).
    """
    n_chunks = math.ceil(audio_seconds / 30.0)
    return n_chunks * 1500  # encoder output tokens per chunk


def print_whisper_overview():
    print("=" * 70)
    print("Whisper Audio Encoder Overview")
    print("=" * 70)
    print(f"  {'Model':12}  {'Params':8}  {'Enc L':6}  {'Dec L':6}  "
          f"{'d_model':8}  {'WER%':6}")
    print("  " + "-" * 58)
    for m in WHISPER_MODELS:
        print(f"  {m.name:12}  {m.params_m:7.0f}M  {m.n_encoder_layers:6}  "
              f"{m.n_decoder_layers:6}  {m.d_model:8}  {m.wer_librispeech:6.1f}")

    print()
    print("  Audio processing pipeline:")
    print("  1. Resample to 16kHz mono")
    print("  2. Compute 80-band log Mel spectrogram: 30s → 3000 frames × 80 bins")
    print("  3. Conv encoder: stride-2 convolutions → 1500 frames × d_model")
    print("  4. Transformer encoder: 1500 × d_model → 1500 encoder outputs")
    print("  5. Decoder cross-attends to encoder outputs, autoregressively generates text")
    print()

    print("  Tokens produced by Whisper encoder for various audio lengths:")
    for secs in [5, 10, 30, 60, 120, 300]:
        tok = whisper_audio_to_tokens(secs)
        chunks = math.ceil(secs / 30)
        print(f"  {secs:4}s  →  {chunks} chunk(s)  →  {tok:,} encoder tokens")
    print()

    print("  Integration with VLMs:")
    print("  Audio features from Whisper encoder feed into an audio projector")
    print("  (similar to vision projector) that maps 1500 × d_whisper → N × d_llm")
    print("  Examples: Qwen-Audio (1500 tokens/30s), InternOmni, MiniCPM-o 3.0")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MULTI-MODAL THROUGHPUT ADVISOR
# ─────────────────────────────────────────────────────────────────────────────

def advise_vlm_deployment(
    vlm: VLMSpec,
    hbm_gb: float,
    img_w: int, img_h: int,
    n_text_tokens: int,
    n_output_tokens: int,
    target_concurrency: int,
) -> None:
    print("=" * 65)
    print("VLM Deployment Advisor")
    print("=" * 65)

    n_vis, n_tiles, tile_desc = visual_tokens_for_image(vlm, img_w, img_h)
    total_input  = n_vis + n_text_tokens
    total_tokens = total_input + n_output_tokens

    # Estimate model weight size (rough: llm 8B + encoder 300M ≈ 8.3B params)
    llm_params_b = (vlm.llm_layers * 4 * vlm.llm_dim * vlm.llm_dim) / 1e9
    enc_params_b = vlm.encoder.params_m / 1e3
    total_params_b = llm_params_b + enc_params_b
    weights_gb   = total_params_b * 2  # BF16

    kv_per_req   = kv_bytes_for_visual_tokens(vlm, total_tokens) / 1e9
    total_kv_gb  = kv_per_req * target_concurrency
    total_needed = weights_gb + total_kv_gb
    usable       = hbm_gb * 0.90
    fits         = total_needed <= usable
    max_conc     = int((usable - weights_gb) / kv_per_req) if kv_per_req > 0 else 0

    print(f"  VLM:         {vlm.name}")
    print(f"  Hardware:    {hbm_gb:.0f} GB GPU")
    print(f"  Image:       {img_w}×{img_h} → {n_vis:,} visual tokens  ({tile_desc})")
    print(f"  Text tokens: {n_text_tokens}")
    print(f"  Output:      {n_output_tokens}")
    print(f"  Total/req:   {total_tokens:,} tokens")
    print()
    print(f"  Weights:     {weights_gb:.1f} GB")
    print(f"  KV/request:  {kv_per_req*1000:.0f} MB")
    print(f"  KV total:    {total_kv_gb*1000:.0f} MB  ({target_concurrency} users)")
    print(f"  Required:    {total_needed:.1f} GB  (usable: {usable:.1f} GB)")
    print()
    if fits:
        print(f"  ✓  Fits. Max concurrency at this resolution: {max_conc}")
    else:
        print(f"  ✗  Exceeds budget. Max feasible concurrency: {max_conc}")
        print(f"     Consider: fewer tiles, smaller encoder (MiniCPM-V), FP8 KV, INT4 weights")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Chapter 29 — Multimodal Inference Demo (Python)")
    print("=" * 70 + "\n")

    print_token_table()
    print_prefill_breakdown()
    print_tiling_table()
    print_whisper_overview()

    advise_vlm_deployment(
        vlm=VLM_FAMILY[1],   # LLaVA-1.6-7B
        hbm_gb=80.0,
        img_w=1024, img_h=768,
        n_text_tokens=512,
        n_output_tokens=256,
        target_concurrency=32,
    )

```

## C++ — `multimodal_demo.cpp`

```cpp
// multimodal_demo.cpp
// Chapter 29 — Multimodal Inference Demo (C++)
//
// Demonstrates without external dependencies:
//   1. Patch extraction geometry and token count
//   2. Vision encoder FLOP estimation
//   3. Tile strategy optimization (aspect-ratio matching)
//   4. KV cache cost for visual tokens
//   5. Memory budget planning for VLMs
//   6. Whisper audio token calculation
//
// Build: g++ -O2 -std=c++17 -o multimodal_demo multimodal_demo.cpp
// Run:   ./multimodal_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

static void print_section(const std::string& title) {
    std::string bar(72, '-');
    std::cout << "\n" << bar << "\n  " << title << "\n" << bar << "\n";
}

static std::string comma(long long n) {
    if (n < 0) return "-" + comma(-n);
    if (n < 1000) return std::to_string(n);
    return comma(n / 1000) + "," + [](long long r){
        char buf[8]; std::snprintf(buf, sizeof(buf), "%03lld", r);
        return std::string(buf);
    }(n % 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// SPECS
// ─────────────────────────────────────────────────────────────────────────────

struct EncoderSpec {
    const char* name;
    int   patch_px;          // patch side in pixels
    int   input_res;         // encoder input resolution
    int   output_dim;        // embedding dimension
    int   n_layers;          // transformer blocks
    double params_m;         // millions of params
};

struct VLMSpec {
    const char* name;
    EncoderSpec enc;
    double compression;      // patch reduction factor (>1 means fewer tokens)
    int    llm_layers;
    int    llm_kv_heads;
    int    llm_head_dim;
    int    llm_dim;
    int    max_tiles;        // content tiles
    bool   thumbnail;        // add global thumbnail tile
    int    max_resolution;   // max input side in pixels
};

static const EncoderSpec CLIP_L  = {"CLIP ViT-L/14@336", 14, 336, 1024, 24, 307};
static const EncoderSpec SIGLIP  = {"SigLIP ViT-SO400M", 14, 384, 1152, 27, 400};
static const EncoderSpec INTERN  = {"InternViT-300M",    14, 448, 1024, 24, 300};

static const VLMSpec VLMS[] = {
    {"LLaVA-1.5-7B", CLIP_L,  1.0, 32, 8, 128, 4096, 1, false, 336},
    {"LLaVA-1.6-7B", CLIP_L,  1.0, 32, 8, 128, 4096, 4, true,  672},
    {"InternVL2-8B", INTERN,  4.0, 32, 8, 128, 4096, 12,true,  4032},
    {"Qwen2-VL-7B",  SIGLIP,  1.0, 28, 4, 128, 3584, 1, false, 2048},
    {"MiniCPM-V-2.6",SIGLIP,  9.0, 32, 8, 128, 4096, 9, true,  1800},
};
static const int N_VLMS = 5;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  PATCH GEOMETRY
// ─────────────────────────────────────────────────────────────────────────────

static int patches_per_tile(const EncoderSpec& e) {
    int n = e.input_res / e.patch_px;
    return n * n;
}

static int tokens_per_tile(const VLMSpec& v) {
    return static_cast<int>(patches_per_tile(v.enc) / v.compression);
}

// Returns (n_content_tiles, total_tiles_with_thumb, visual_tokens)
static std::tuple<int,int,int> visual_tokens(const VLMSpec& v, int img_w, int img_h) {
    if (v.max_tiles == 1) {
        int total = 1 + (v.thumbnail ? 1 : 0);
        return {1, total, total * tokens_per_tile(v)};
    }
    int tw = std::max(1, (int)std::round((double)img_w / v.enc.input_res));
    int th = std::max(1, (int)std::round((double)img_h / v.enc.input_res));
    int content = std::min(tw * th, v.max_tiles);
    int total   = content + (v.thumbnail ? 1 : 0);
    return {content, total, total * tokens_per_tile(v)};
}

static long long kv_bytes(const VLMSpec& v, int n_tokens, int dtype_bytes = 2) {
    return 2LL * v.llm_layers * v.llm_kv_heads * v.llm_head_dim * n_tokens * dtype_bytes;
}

static void demo_patch_geometry() {
    print_section("Patch Geometry and Visual Token Count");

    struct Img { int w, h; const char* desc; };
    const Img images[] = {
        {336,  336,  "Square thumbnail"},
        {672,  672,  "Standard HD"},
        {1024, 768,  "Landscape photo"},
        {768,  1024, "Portrait photo"},
        {1920, 1080, "Full HD screenshot"},
    };
    const int N_IMG = 5;

    for (int ii = 0; ii < N_IMG; ++ii) {
        auto& img = images[ii];
        std::cout << "\n  Image " << img.w << "×" << img.h << "  (" << img.desc << ")\n";
        std::cout << "  " << std::left << std::setw(22) << "Model"
                  << std::right << std::setw(9) << "Tokens"
                  << std::setw(12) << "KV (MB)"
                  << std::setw(8) << "Tiles"
                  << "\n  " << std::string(51, '-') << "\n";

        for (int vi = 0; vi < N_VLMS; ++vi) {
            auto& v = VLMS[vi];
            auto [content, total, ntok] = visual_tokens(v, img.w, img.h);
            double kv_mb = kv_bytes(v, ntok) / 1e6;
            std::cout << "  " << std::left << std::setw(22) << v.name
                      << std::right << std::setw(9) << comma(ntok)
                      << std::setw(10) << std::fixed << std::setprecision(1) << kv_mb << " MB"
                      << std::setw(8) << total
                      << "\n";
        }
    }
    std::cout << "\n";

    // Assertions
    // LLaVA-1.5: always 576 tokens (1 tile, no thumb, compression=1)
    auto [c1, t1, n1] = visual_tokens(VLMS[0], 1024, 768);
    assert(n1 == 576);
    std::cout << "  [ASSERT] LLaVA-1.5 always produces 576 tokens: " << n1 << " ✓\n";

    // InternVL2: compression=4, so 448/14=32 → 1024 patches / 4 = 256 per tile
    assert(tokens_per_tile(VLMS[2]) == 256);
    std::cout << "  [ASSERT] InternVL2 tokens/tile = 256 (1024 patches ÷ 4): ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  VISION ENCODER FLOP ESTIMATION
// ─────────────────────────────────────────────────────────────────────────────

static double encoder_flops(const EncoderSpec& e) {
    // Per transformer block: ~12 × d² × N FLOPs (self-attn + FFN)
    int N = patches_per_tile(e);
    return (double)e.n_layers * 12.0 * (double)(e.output_dim * e.output_dim) * N;
}

static void demo_encoder_flops() {
    print_section("Vision Encoder FLOP Budget");

    const EncoderSpec encoders[] = {CLIP_L, SIGLIP, INTERN};
    const int N = 3;
    const double H100_TFLOPS = 989.0;

    std::cout << "\n  " << std::left << std::setw(24) << "Encoder"
              << std::right << std::setw(14) << "FLOPs"
              << std::setw(14) << "Time (H100)"
              << std::setw(12) << "Patches"
              << "\n  " << std::string(64, '-') << "\n";

    for (int i = 0; i < N; ++i) {
        auto& e = encoders[i];
        double flops = encoder_flops(e);
        double ms    = flops / (H100_TFLOPS * 1e12) * 1e3;
        int    pats  = patches_per_tile(e);
        std::cout << "  " << std::left << std::setw(24) << e.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2)
                  << flops / 1e9 << " GF"
                  << std::setw(12) << std::setprecision(3) << ms << " ms"
                  << std::setw(12) << pats
                  << "\n";
    }

    // Multi-tile: 4 tiles + 1 thumbnail = 5 passes
    double total_5tile = 5 * encoder_flops(CLIP_L);
    double ms_5tile    = total_5tile / (H100_TFLOPS * 1e12) * 1e3;
    std::cout << "\n  LLaVA-1.6 HD (5 CLIP passes): "
              << std::setprecision(2) << ms_5tile << " ms on H100\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  OPTIMAL TILING
// ─────────────────────────────────────────────────────────────────────────────

static std::tuple<int,int,int> best_tiling(int img_w, int img_h,
                                            int tile_px, int max_tiles) {
    double aspect = (double)img_w / img_h;
    int best_tw = 1, best_th = 1;
    double best_score = 1e18;

    for (int tw = 1; tw <= max_tiles; ++tw)
        for (int th = 1; th <= max_tiles; ++th) {
            if (tw * th > max_tiles) continue;
            double score = std::abs((double)tw / th - aspect);
            if (score < best_score) {
                best_score = score;
                best_tw = tw; best_th = th;
            }
        }
    return {best_tw, best_th, best_tw * best_th};
}

static void demo_tiling() {
    print_section("Optimal Tile Selection (LLaVA-1.6, max_tiles=4, tile=336px)");

    struct Img { int w, h; };
    const Img imgs[] = {
        {336, 336}, {672, 336}, {336, 672},
        {1024, 576}, {576, 1024}, {800, 600}, {1920, 1080},
    };
    const int N = 7;

    std::cout << "\n  " << std::left << std::setw(14) << "Image"
              << std::right << std::setw(10) << "Aspect"
              << std::setw(10) << "Tiling"
              << std::setw(8) << "Tiles"
              << std::setw(20) << "Tokens (w/ thumb)"
              << "\n  " << std::string(62, '-') << "\n";

    for (int i = 0; i < N; ++i) {
        auto& img = imgs[i];
        auto [tw, th, n] = best_tiling(img.w, img.h, 336, 4);
        double aspect = (double)img.w / img.h;
        int tokens = (n + 1) * 576;  // +1 for thumbnail

        std::ostringstream res, til;
        res << img.w << "×" << img.h;
        til << tw << "×" << th;

        std::cout << "  " << std::left << std::setw(14) << res.str()
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << aspect
                  << std::setw(10) << til.str()
                  << std::setw(8) << n
                  << std::setw(20) << comma(tokens)
                  << "\n";
    }

    // Assert: square image tiles as 1×1 or 2×2
    auto [tw_sq, th_sq, n_sq] = best_tiling(672, 672, 336, 4);
    assert(tw_sq == th_sq);  // aspect 1.0 → square tiling
    std::cout << "\n  [ASSERT] Square image → square tiling (" << tw_sq << "×" << th_sq << "): ✓\n";

    // Assert: landscape 16:9 tiles as 2×1
    auto [tw_ls, th_ls, n_ls] = best_tiling(1920, 1080, 336, 4);
    assert(tw_ls > th_ls);  // wider than tall
    std::cout << "  [ASSERT] Landscape image → wider tiling (" << tw_ls << "×" << th_ls << "): ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  KV CACHE BUDGET FOR VLMs
// ─────────────────────────────────────────────────────────────────────────────

static void demo_kv_budget() {
    print_section("KV Cache Budget — H100 80GB VLM Deployment");

    const double HBM_GB      = 80.0;
    const double OVERHEAD    = 0.10;
    const double USABLE      = HBM_GB * (1.0 - OVERHEAD);

    // Rough weight estimates: 8B model BF16 ≈ 16GB + 300-400M encoder ≈ 0.6GB
    const double WEIGHTS_GB  = 16.6;
    const double AVAIL_KV_GB = USABLE - WEIGHTS_GB;

    // Request profile: 1 image (1024×768) + 512 text + 256 output
    const int IMG_W = 1024, IMG_H = 768;
    const int TEXT_TOK = 512, OUT_TOK = 256;

    std::cout << "\n  H100 usable HBM: " << USABLE << " GB\n"
              << "  Weights (LLM + encoder BF16): " << WEIGHTS_GB << " GB\n"
              << "  Available for KV: " << AVAIL_KV_GB << " GB\n\n";

    std::cout << "  " << std::left << std::setw(22) << "VLM"
              << std::right << std::setw(10) << "Vis tok"
              << std::setw(12) << "KV/req(MB)"
              << std::setw(14) << "Max concurr."
              << "\n  " << std::string(58, '-') << "\n";

    for (int vi = 0; vi < N_VLMS; ++vi) {
        auto& v = VLMS[vi];
        auto [c, t, n_vis] = visual_tokens(v, IMG_W, IMG_H);
        int total_tok = n_vis + TEXT_TOK + OUT_TOK;
        double kv_mb  = kv_bytes(v, total_tok) / 1e6;
        int max_conc  = static_cast<int>(AVAIL_KV_GB * 1e3 / kv_mb);

        std::cout << "  " << std::left << std::setw(22) << v.name
                  << std::right << std::setw(10) << comma(n_vis)
                  << std::setw(12) << std::fixed << std::setprecision(1) << kv_mb
                  << std::setw(14) << max_conc
                  << "\n";
    }
    std::cout << "\n  Request profile: 1024×768 image + 512 text + 256 output tokens\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  WHISPER AUDIO TOKEN CALCULATION
// ─────────────────────────────────────────────────────────────────────────────

struct WhisperSpec {
    const char* name;
    int   enc_layers;
    int   dec_layers;
    int   d_model;
    double params_m;
    double wer_clean;
};

static const WhisperSpec WHISPER[] = {
    {"tiny",     4,  4,  384,  39,  5.7},
    {"base",     6,  6,  512,  74,  4.2},
    {"small",    12, 12, 768,  244, 3.0},
    {"medium",   24, 24, 1024, 769, 2.0},
    {"large-v3", 32, 32, 1280, 1550,2.0},
    {"turbo",    32, 2,  1280, 809, 2.7},
};
static const int N_WHISPER = 6;

static int whisper_encoder_tokens(double audio_s) {
    // 30s chunk → 1500 encoder tokens
    int chunks = static_cast<int>(std::ceil(audio_s / 30.0));
    return chunks * 1500;
}

static double whisper_encode_ms(const WhisperSpec& w, double audio_s,
                                  double H100_TFLOPS = 989.0) {
    int chunks = static_cast<int>(std::ceil(audio_s / 30.0));
    // Each 30s chunk: 3000 Mel frames → 1500 after conv
    // Per block: ~12 × d² × 1500 FLOPs
    double flops_per_chunk = w.enc_layers * 12.0 * (double)(w.d_model * w.d_model) * 1500;
    double total_flops = chunks * flops_per_chunk;
    return total_flops / (H100_TFLOPS * 1e12) * 1e3;
}

static void demo_whisper() {
    print_section("Whisper Audio Encoder — Token Count and Compute");

    std::cout << "\n  Model specs:\n";
    std::cout << "  " << std::left << std::setw(12) << "Model"
              << std::right << std::setw(10) << "Params"
              << std::setw(10) << "Enc L"
              << std::setw(10) << "Dec L"
              << std::setw(10) << "d_model"
              << std::setw(10) << "WER%"
              << "\n  " << std::string(62, '-') << "\n";
    for (int i = 0; i < N_WHISPER; ++i) {
        auto& m = WHISPER[i];
        std::cout << "  " << std::left << std::setw(12) << m.name
                  << std::right << std::setw(8) << std::fixed << std::setprecision(0) << m.params_m << "M"
                  << std::setw(10) << m.enc_layers
                  << std::setw(10) << m.dec_layers
                  << std::setw(10) << m.d_model
                  << std::setw(10) << std::setprecision(1) << m.wer_clean
                  << "\n";
    }

    std::cout << "\n  Audio → encoder tokens and encode time (large-v3, H100):\n";
    std::cout << "  " << std::right << std::setw(10) << "Duration"
              << std::setw(10) << "Chunks"
              << std::setw(14) << "Enc tokens"
              << std::setw(14) << "Encode ms"
              << "\n  " << std::string(48, '-') << "\n";

    const double durations[] = {5, 10, 30, 60, 120, 300};
    const int ND = 6;
    for (int i = 0; i < ND; ++i) {
        double secs = durations[i];
        int chunks  = static_cast<int>(std::ceil(secs / 30.0));
        int enc_tok = whisper_encoder_tokens(secs);
        double ms   = whisper_encode_ms(WHISPER[4], secs);  // large-v3
        std::cout << "  " << std::right << std::setw(8) << std::fixed << std::setprecision(0) << secs << "s"
                  << std::setw(10) << chunks
                  << std::setw(14) << comma(enc_tok)
                  << std::setw(14) << std::setprecision(2) << ms << " ms"
                  << "\n";
    }

    // Assert: 30s → exactly 1500 encoder tokens
    assert(whisper_encoder_tokens(30.0) == 1500);
    std::cout << "\n  [ASSERT] 30s audio → 1500 encoder tokens: ✓\n";

    // Assert: encode time < 100ms for 30s chunk on H100
    double ms_30s = whisper_encode_ms(WHISPER[4], 30.0);
    assert(ms_30s < 100.0);
    std::cout << "  [ASSERT] Encode 30s < 100ms on H100: "
              << std::setprecision(2) << ms_30s << "ms ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << "  Chapter 29 — Multimodal Inference Demo (C++)\n"
              << std::string(72, '=') << "\n";

    demo_patch_geometry();
    demo_encoder_flops();
    demo_tiling();
    demo_kv_budget();
    demo_whisper();

    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  All demos complete.\n"
              << std::string(72, '=') << "\n\n";
    return 0;
}

```

