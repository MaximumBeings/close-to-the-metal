# Chapter 22: LoRA Serving and Adapter Hot-Swapping

> *"The insight behind LoRA is deceptively simple: during fine-tuning, weight updates live in a low-rank subspace. If that is true, you do not need to store a full copy of the model for every fine-tune — you only need to store the rank-r delta."*

---

## What You Will Understand

- What LoRA is, derived from first principles: why the low-rank assumption holds, and what rank r means geometrically
- How to calculate the exact memory cost of a LoRA adapter for any model
- How vLLM routes each inference request to a different adapter without reloading weights
- How to serve 100+ fine-tuned variants from a single base model in production
- How llama.cpp handles LoRA differently — load-time merge rather than runtime injection
- The full fine-tune → convert → register → serve pipeline

**What you need first:** Chapter 2 (weight memory), Chapter 6 (KV cache and block management — the same slot management concepts apply to adapter memory), Chapter 8 (startup and weight loading).

---

## §22.1  The LoRA Intuition

Suppose you have a 7B-parameter base model and you want to create 50 domain-specific variants: one for legal contracts, one for medical notes, one for customer service, one per language, and so on. The naïve approach:

```
  50 fine-tuned models × 14 GB each (7B @ BF16)  =  700 GB of weights
  Serving all simultaneously on H100s:  700 GB ÷ 80 GB/GPU  ≈  9 H100s
  Cost:  9 × $3.20/hr  =  $28.80/hr  just for weights
```

LoRA (Low-Rank Adaptation, Hu et al. 2021) cuts this dramatically by observing: **the weight updates that arise during fine-tuning occupy a low-dimensional subspace**.

If a weight matrix W ∈ ℝ^(d_out × d_in) is updated by ΔW during fine-tuning, and if ΔW has low rank, then:

```
  ΔW  ≈  B · A        where B ∈ ℝ^(d_out × r),   A ∈ ℝ^(r × d_in)
                              r << min(d_out, d_in)
```

Rather than storing all of ΔW (d_out × d_in floats), you store only A and B (r × (d_in + d_out) floats). For r = 16 and a 4096×4096 attention weight:

```
WORKED EXAMPLE 22.1 — LoRA memory saving for one weight matrix
───────────────────────────────────────────────────────────────
Given:
  d_out = 4096,  d_in = 4096,  r = 16,  dtype = BF16 (2 bytes)

Full ΔW:
  4096 × 4096 × 2 bytes  =  33,554,432 bytes  =  32 MB

LoRA A + B:
  A:  r × d_in  =  16 × 4096  =  65,536 parameters
  B:  d_out × r  =  4096 × 16  =  65,536 parameters
  Total:  131,072 × 2 bytes  =  262,144 bytes  =  256 KB

Compression ratio:  32 MB ÷ 256 KB  =  128×
───────────────────────────────────────────────────────────────
```

128× compression **per weight matrix**. A full adapter (covering Q, K, V, O projections across all layers) is much larger but still tiny compared to the base model.

---

## §22.2  LoRA Mathematics — Step by Step

### 22.2.1  The Forward Pass with LoRA

During inference with a LoRA adapter, the modified weight matrix is:

```
  W_effective  =  W_base  +  (B · A) · scaling_factor

  where  scaling_factor  =  alpha / r
         alpha  =  hyperparameter (often = r, making the factor = 1)
```

The forward pass becomes:

```
  output  =  x · W_base^T  +  x · A^T · B^T · (alpha/r)
           =  x · (W_base + B·A·(alpha/r))^T
```

This means at inference time you can either:

1. **Merge**: compute `W_merged = W_base + B·A·(alpha/r)` once and use `W_merged` — zero overhead per forward pass, but you need one copy of W per adapter
2. **Inject**: keep `W_base` as-is and add the LoRA term in a fused kernel — more arithmetic per forward pass, but one copy of W serves all adapters

vLLM uses approach 2 (injection). llama.cpp defaults to approach 1 (merge at load time).

### 22.2.2  Why Low Rank Works

```
  The intuition:

  Full fine-tuning updates every dimension of the weight space.
  But task adaptation rarely needs all dimensions — a legal fine-tune
  shifts the model's "legal vocabulary" and "citation style" subspace,
  not its "world-knowledge" subspace.

  Rank-r captures the r most important directions of change.
  Empirically, r=8 to r=64 recovers most of the task improvement,
  while r=256 approaches full fine-tuning quality.

  ┌───────────────────────────────────────────────────┐
  │  Quality vs. adapter size (7B model, legal task)  │
  │                                                   │
  │  r     Adapter MB    Task score    vs. full FT    │
  │  4       8 MB          78.2           -5.1%       │
  │  8      16 MB          81.4           -2.9%       │
  │  16     32 MB          83.1           -1.1%       │
  │  64    128 MB          83.8           -0.4%       │
  │  Full FT:  14,000 MB   84.2           baseline    │
  └───────────────────────────────────────────────────┘
```

---

## §22.3  Full Adapter Memory Budget

For a complete LLM (not just one weight matrix), LoRA adapters are typically applied to the attention projections: Q, K, V, and O. Some recipes also include the MLP up/down projections.

```
WORKED EXAMPLE 22.2 — Full adapter budget for Llama 3.1 8B
───────────────────────────────────────────────────────────
Given:
  n_layers = 32,  d_model = 4096,  r = 16,  alpha = 16
  Target matrices: Q, K, V, O  (4 per layer)
  dtype = BF16 (2 bytes)

Per matrix per layer:
  A: r × d_in  = 16 × 4096 = 65,536 params
  B: d_out × r = 4096 × 16 = 65,536 params
  Per matrix: 131,072 × 2 bytes = 256 KB

Per layer (4 matrices):
  4 × 256 KB = 1 MB

Full adapter (32 layers, attention only):
  32 × 1 MB = 32 MB

Including MLP projections (up + down, d_ff = 14,336):
  up:   A = 16 × 4096 = 65,536;  B = 14,336 × 16 = 229,376  →  590 KB
  down: A = 16 × 14,336 = 229,376;  B = 4096 × 16 = 65,536  →  590 KB
  Per layer: 2 × 590 KB = 1.18 MB
  32 layers: ~37.7 MB

Total (attention + MLP, r=16, 32 layers):
  ~32 MB + ~37.7 MB  ≈  70 MB

Compare to base model:
  Llama 3.1 8B @ BF16:  ~16,000 MB (16 GB)
  Adapter:  70 MB  ≈  0.4% of base model size
───────────────────────────────────────────────────────────
```

At 70 MB per adapter, an H100 with 80 GB of HBM can hold the base model (16 GB) plus hundreds of adapters simultaneously:

```
  Available for adapters:  (80 GB × 0.90 utilization) − 16 GB − 1.5 GB activations
                         = 72 GB − 17.5 GB
                         = 54.5 GB

  Adapters at 70 MB each:   54.5 GB ÷ 0.07 GB  ≈  778 adapters
```

In practice vLLM recommends loading a subset of "hot" adapters and swapping cold ones — but the theoretical capacity is large.

---

## §22.4  vLLM LoRA Architecture

### 22.4.1  Enabling Multi-LoRA Serving

```bash
# Key flag choices:
#   --max-loras 4              max adapters loaded simultaneously in HBM
#   --max-lora-rank 64         maximum r across all adapters
#   --lora-extra-vocab-size 0  for adapters with custom vocab tokens
#   --max-cpu-loras 32         LRU cache of adapters in CPU RAM
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --lora-extra-vocab-size 0 \
    --max-cpu-loras 32
```

`--max-loras` controls **hot slots** — how many adapters are resident in HBM at once. `--max-cpu-loras` controls the warm LRU cache in CPU RAM. Adapters not in either are fetched from disk.

### 22.4.2  Per-Request Adapter Routing

Each API request specifies which adapter to use via the `model` field or a custom `lora_request` extension:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=4,
    max_lora_rank=64,
)

params = SamplingParams(temperature=0.7, max_tokens=256)

# Three requests, three different adapters
outputs = llm.generate(
    [
        "Draft a non-disclosure agreement for...",
        "Summarize this patient intake form...",
        "Respond to this customer complaint about...",
    ],
    params,
    lora_request=[
        LoRARequest("legal-v2",    lora_int_id=1, lora_path="/adapters/legal-v2"),
        LoRARequest("medical-v1",  lora_int_id=2, lora_path="/adapters/medical-v1"),
        LoRARequest("support-v3",  lora_int_id=3, lora_path="/adapters/support-v3"),
    ],
)
```

### 22.4.3  What Happens Inside vLLM During LoRA Routing

```
  Request arrives → scheduler assigns to a sequence →
  LoRA manager checks if adapter is in HBM hot slot:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  ADAPTER MANAGER                                                    │
  │                                                                     │
  │  Hot slots (HBM, max_loras=4):                                      │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
  │  │ legal-v2 │  │medical-v1│  │support-v3│  │ (empty)  │           │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
  │                                                                     │
  │  Warm cache (CPU, max_cpu_loras=32):                                │
  │  legal-v1, medical-v2, es-support-v1, fr-support-v1, ...           │
  │                                                                     │
  │  On cache miss:                                                     │
  │  1. Evict LRU from hot slots (if all 4 full)                        │
  │  2. Copy evicted adapter to CPU warm cache                          │
  │  3. Load new adapter from CPU (or disk) → HBM hot slot             │
  └─────────────────────────────────────────────────────────────────────┘

  During forward pass (per layer, per request):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  output = x @ W_base.T + (x @ A.T @ B.T) * (alpha/r)              │
  │                           ↑ LoRA term injected by fused CUDA kernel │
  │                                                                     │
  │  Requests with the same base model but different adapters are        │
  │  batched together for the base model computation, then each gets    │
  │  its own LoRA term added via a grouped GEMM.                        │
  └─────────────────────────────────────────────────────────────────────┘
```

The key efficiency: **base model weights are read once** from HBM, shared across all requests in the batch. The LoRA delta computation is small (r << d) and added per-request. GPU utilization stays high.

### 22.4.4  Memory Layout

```
  HBM layout with multi-LoRA enabled:

  ┌─────────────────────────────────────────────────────┐
  │  Region 1: Base model weights  (16 GB for 8B BF16)  │
  │  ─────────────────────────────────────────────────  │
  │  Region 2: LoRA hot slots  (max_loras × adapter_mb) │
  │            4 × 70 MB = 280 MB                       │
  │  ─────────────────────────────────────────────────  │
  │  Region 3: KV cache blocks  (remainder)             │
  │            72 GB − 16 GB − 0.28 GB − 1.5 GB ≈ 54 GB│
  └─────────────────────────────────────────────────────┘
```

LoRA hot slots take only ~280 MB for 4 adapters — negligible compared to base weights and KV cache.

---

## §22.5  llama.cpp LoRA — Load-Time Merge

llama.cpp's approach differs fundamentally: instead of injecting the LoRA delta at inference time, it **merges** the adapter into the base weights at load time.

```cpp
// llama.cpp LoRA merge at startup
llama_model_params model_params = llama_model_default_params();
llama_model* model = llama_model_load_from_file(
    "./Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    model_params
);

// Apply LoRA adapter — this modifies model's weight tensors in-place
int result = llama_model_apply_lora_from_file(
    model,
    "./legal-v2-lora.gguf",   // adapter file in GGUF format
    1.0f,                      // scale = alpha/r (pre-computed or 1.0)
    nullptr,                   // base model path (for differential scaling)
    4                          // n_threads for merge computation
);
```

**What happens during `llama_model_apply_lora_from_file`:**
1. Loads adapter GGUF file (small, typically < 100 MB)
2. For each weight tensor in the model that has a corresponding LoRA delta:
   - Dequantizes the base weight tensor to FP32
   - Computes B·A (matrix multiply, in FP32)
   - Scales by alpha/r
   - Adds to the dequantized base weight
   - Re-quantizes the result
3. The model is now the merged variant; the adapter tensors are discarded

**Trade-offs vs. vLLM's injection approach:**

```
  Aspect               │ vLLM injection          │ llama.cpp merge
  ─────────────────────┼─────────────────────────┼──────────────────────────
  Startup cost         │ Zero (adapters loaded    │ O(n_layers × d²) merge
                       │  lazily on demand)       │  computation at load
  Per-request overhead │ Small (fused LoRA GEMM)  │ Zero (merged in weights)
  Multiple adapters    │ Yes (hot-swap mid-batch) │ No (one merged model)
  Memory               │ Base + N × adapter_MB    │ Base only (adapter freed)
  Quality              │ Exact (no re-quant)      │ May degrade with Q4 merge
```

`[COMMON TRAP]` Merging a LoRA adapter into a quantized model (Q4_K_M) introduces quantization error twice: once when the base was quantized, and again when the merged result is re-quantized. For high-precision tasks, merge into an FP16 base first, then re-quantize the result.

---

## §22.6  The Fine-Tune → Serve Pipeline

The end-to-end workflow for getting a LoRA fine-tune into production:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 1: Fine-tune with PEFT / Axolotl / LLaMA-Factory             │
  │                                                                     │
  │  from peft import LoraConfig, get_peft_model                        │
  │  config = LoraConfig(r=16, lora_alpha=16,                           │
  │                      target_modules=["q_proj","v_proj"],            │
  │                      lora_dropout=0.05, bias="none")                │
  │  model = get_peft_model(base_model, config)                         │
  │  # ... training loop ...                                            │
  │  model.save_pretrained("./legal-v2-adapter")                        │
  │  # Saves: adapter_config.json, adapter_model.safetensors            │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────────┐
  │  STEP 2: Validate adapter quality                                   │
  │                                                                     │
  │  # Quick eval before deploying                                      │
  │  from peft import PeftModel                                         │
  │  model = PeftModel.from_pretrained(base_model, "./legal-v2-adapter")│
  │  # Run evaluation set, check task metrics                           │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────────┐
  │  STEP 3: Register with vLLM (no conversion needed for HF adapters)  │
  │                                                                     │
  │  vllm serve meta-llama/Llama-3.1-8B-Instruct                       │
  │      --enable-lora                                                  │
  │      --max-loras 8                                                  │
  │      # Adapters are loaded on first request, no pre-registration    │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────────┐
  │  STEP 4: For llama.cpp — convert adapter to GGUF                   │
  │                                                                     │
  │  python llama.cpp/convert_lora_to_gguf.py \                         │
  │      --base ./Llama-3.1-8B-Instruct \                               │
  │      --lora ./legal-v2-adapter \                                    │
  │      --outfile ./legal-v2-lora.gguf                                 │
  └─────────────────────────────────────────────────────────────────────┘
```

### 22.6.1  Adapter Registry Pattern

For production with many adapters, maintain an adapter registry rather than hardcoding paths:

```python
# §22.6.1 — Adapter registry for dynamic routing
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class AdapterSpec:
    name:       str
    version:    str
    lora_path:  str
    lora_int_id: int       # vLLM requires a unique integer ID per adapter
    rank:       int
    description: str

class AdapterRegistry:
    def __init__(self, registry_path: str):
        with open(registry_path) as f:
            raw = json.load(f)
        self._adapters = {
            spec["name"]: AdapterSpec(**spec)
            for spec in raw["adapters"]
        }
        self._next_id = max(s.lora_int_id for s in self._adapters.values()) + 1

    def get(self, name: str) -> AdapterSpec:
        if name not in self._adapters:
            raise KeyError(f"Unknown adapter: {name}")
        return self._adapters[name]

    def register(self, spec: AdapterSpec) -> None:
        """Hot-register a new adapter without restarting vLLM."""
        self._adapters[spec.name] = spec

    def list_adapters(self) -> list[str]:
        return sorted(self._adapters.keys())

# registry.json example:
# {
#   "adapters": [
#     {"name": "legal-v2", "version": "2.1.0",
#      "lora_path": "/adapters/legal-v2", "lora_int_id": 1,
#      "rank": 16, "description": "Legal contracts EN"},
#     ...
#   ]
# }
```

---

## §22.7  A/B Testing Adapters in Production

With per-request adapter routing, A/B testing fine-tunes becomes trivial:

```python
# §22.7 — A/B test routing: 90% traffic to stable, 10% to candidate
import random
from vllm.lora.request import LoRARequest

def get_lora_request(user_id: str) -> LoRARequest:
    """Route 10% of users to the candidate adapter."""
    bucket = int(user_id, 16) % 100 if all(c in "0123456789abcdef"
                                            for c in user_id[-4:]) \
             else hash(user_id) % 100

    if bucket < 10:
        return LoRARequest("legal-v3-candidate", lora_int_id=10,
                           lora_path="/adapters/legal-v3-candidate")
    else:
        return LoRARequest("legal-v2-stable", lora_int_id=1,
                           lora_path="/adapters/legal-v2")
```

Log the `lora_request.lora_name` in every response and correlate with downstream quality metrics (human ratings, task completion rates) to make the rollout decision.

---

## §22.8  Code

### Python: Full Multi-LoRA Serving Demo

```python
#!/usr/bin/env python3
"""
Chapter 22 — Python: Multi-LoRA Serving Demo
=============================================
Demonstrates:
  - LoRA memory budget calculation
  - Adapter registry with per-request routing
  - vLLM LoRARequest integration (mock, no GPU required)
  - A/B test routing logic
"""

from dataclasses import dataclass
from typing import Optional
import random, hashlib, json

# ─── Memory budget calculator ─────────────────────────────────────────────────

PRECISION_BYTES = {"FP32": 4.0, "BF16": 2.0, "FP16": 2.0, "INT8": 1.0}

@dataclass
class LoRAConfig:
    rank:       int           # r
    alpha:      int           # alpha (scaling = alpha / rank)
    target_modules: list[str] # e.g., ["q_proj", "v_proj", "k_proj", "o_proj"]
    dtype:      str = "BF16"

@dataclass
class ModelArchitecture:
    name:       str
    n_layers:   int
    d_model:    int           # hidden dim
    d_ff:       int           # MLP intermediate dim

MODELS = {
    "Llama-3.1-8B":  ModelArchitecture("Llama 3.1 8B",  32, 4096, 14336),
    "Llama-3.3-70B": ModelArchitecture("Llama 3.3 70B", 80, 8192, 28672),
    "Qwen2.5-7B":    ModelArchitecture("Qwen 2.5 7B",   28, 3584, 18944),
}

MODULE_DIMS = {
    # Each maps to (d_out, d_in) for W ∈ ℝ^{d_out × d_in}
    "q_proj": lambda d, _: (d, d),
    "k_proj": lambda d, _: (d, d),   # simplified; GQA would be smaller
    "v_proj": lambda d, _: (d, d),
    "o_proj": lambda d, _: (d, d),
    "gate_proj": lambda d, f: (f, d),
    "up_proj":   lambda d, f: (f, d),
    "down_proj": lambda d, f: (d, f),
}

def adapter_memory_bytes(
    arch: ModelArchitecture,
    config: LoRAConfig,
) -> dict:
    """
    Calculate total LoRA adapter memory.
    Returns breakdown dict with bytes per module and total.
    """
    bpe = PRECISION_BYTES[config.dtype]
    r   = config.rank
    breakdown = {}

    per_layer_bytes = 0
    for mod in config.target_modules:
        if mod not in MODULE_DIMS:
            continue
        d_out, d_in = MODULE_DIMS[mod](arch.d_model, arch.d_ff)
        A_params = r * d_in
        B_params = d_out * r
        mod_bytes = (A_params + B_params) * bpe
        breakdown[mod] = mod_bytes
        per_layer_bytes += mod_bytes

    total_bytes = per_layer_bytes * arch.n_layers
    breakdown["_per_layer"] = per_layer_bytes
    breakdown["_total"]     = total_bytes
    return breakdown


def print_adapter_budgets():
    print("\n" + "=" * 70)
    print("  LoRA Adapter Memory Budget")
    print("=" * 70)

    configs = [
        LoRAConfig(rank=8,  alpha=8,  target_modules=["q_proj","v_proj"]),
        LoRAConfig(rank=16, alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"]),
        LoRAConfig(rank=64, alpha=64, target_modules=["q_proj","k_proj","v_proj","o_proj",
                                                       "gate_proj","up_proj","down_proj"]),
    ]
    mod_labels = ["q+v", "attn×4", "all×7"]

    print(f"  {'Model':<20} {'Modules':<10} {'Rank':>6} {'Adapter MB':>12} "
          f"{'Adapters/H100':>15}")
    print(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*12} {'-'*15}")

    for arch in MODELS.values():
        for cfg, label in zip(configs, mod_labels):
            mem = adapter_memory_bytes(arch, cfg)
            total_mb = mem["_total"] / 1e6
            # H100 80GB: 80×0.90 − weight_gb − 1.5 activations
            base_gb  = {"Llama 3.1 8B": 16, "Llama 3.3 70B": 140,
                        "Qwen 2.5 7B": 14}[arch.name]
            avail_mb = (80*0.90 - base_gb - 1.5) * 1024
            n_adapters = int(avail_mb / total_mb) if total_mb > 0 else 0
            print(f"  {arch.name:<20} {label:<10} {cfg.rank:>6} "
                  f"{total_mb:>11.1f}M {n_adapters:>15,}")
        print()


# ─── Adapter registry ─────────────────────────────────────────────────────────

@dataclass
class AdapterSpec:
    name:        str
    lora_int_id: int
    lora_path:   str
    rank:        int
    description: str
    traffic_pct: float = 100.0   # percentage of traffic routed here

class AdapterRegistry:
    def __init__(self):
        self._adapters: dict[str, AdapterSpec] = {}
        self._routing_groups: dict[str, list[AdapterSpec]] = {}

    def register(self, spec: AdapterSpec) -> None:
        self._adapters[spec.name] = spec

    def get(self, name: str) -> AdapterSpec:
        return self._adapters[name]

    def register_ab_group(self, group_name: str, specs: list[AdapterSpec]) -> None:
        """Register an A/B group. Traffic split by spec.traffic_pct (must sum to 100)."""
        assert abs(sum(s.traffic_pct for s in specs) - 100.0) < 0.01, \
            "Traffic percentages must sum to 100"
        self._routing_groups[group_name] = specs
        for spec in specs:
            self.register(spec)

    def route_ab(self, group_name: str, user_id: str) -> AdapterSpec:
        """Deterministically route user to an adapter in the A/B group."""
        specs = self._routing_groups[group_name]
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        cumulative = 0.0
        for spec in specs:
            cumulative += spec.traffic_pct
            if bucket < cumulative:
                return spec
        return specs[-1]   # fallback


def demo_registry():
    print("\n" + "=" * 70)
    print("  Adapter Registry and A/B Routing Demo")
    print("=" * 70)

    registry = AdapterRegistry()
    registry.register_ab_group("legal-routing", [
        AdapterSpec("legal-v2-stable",    lora_int_id=1,
                    lora_path="/adapters/legal-v2",    rank=16,
                    description="Legal EN stable",     traffic_pct=90.0),
        AdapterSpec("legal-v3-candidate", lora_int_id=10,
                    lora_path="/adapters/legal-v3",    rank=16,
                    description="Legal EN candidate",  traffic_pct=10.0),
    ])

    # Simulate 20 users
    counts = {"legal-v2-stable": 0, "legal-v3-candidate": 0}
    for i in range(200):
        user_id = f"user_{i:04d}"
        spec    = registry.route_ab("legal-routing", user_id)
        counts[spec.name] += 1

    print(f"  Simulated routing for 200 users:")
    for name, count in counts.items():
        pct = count / 200 * 100
        bar = "█" * (count // 5)
        print(f"    {name:<30} {count:>4}  ({pct:4.1f}%)  {bar}")


if __name__ == "__main__":
    print("\nChapter 22 — LoRA Serving Demo")
    print("=" * 70)
    print_adapter_budgets()
    demo_registry()
```

### C++: LoRA Memory Budget Calculator and Merge Walkthrough

```cpp
/**
 * Chapter 22 — C++ Companion: LoRA Memory and Merge Analysis
 * ===========================================================
 * Demonstrates:
 *   - LoRA A and B matrix size calculation for any (d_out, d_in, r)
 *   - Full adapter memory budget for Llama 3.1 8B
 *   - Conceptual merge walkthrough (B·A computed for a tiny example)
 *   - Compression ratio analysis
 *
 * Build:  g++ -std=c++17 -O2 lora_demo.cpp -o lora_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct LoRAConfig {
    int rank;
    int alpha;
};

struct ModuleSpec {
    std::string name;
    int d_out;
    int d_in;
};

struct ModelArch {
    std::string name;
    int n_layers;
    int d_model;
    int d_ff;
};

// ─────────────────────────────────────────────────────────────────────────────
// Memory budget calculations
// ─────────────────────────────────────────────────────────────────────────────

struct AdapterBreakdown {
    std::string module_name;
    long long A_params;
    long long B_params;
    double bytes_mb;
};

static double BPE_BF16 = 2.0;

AdapterBreakdown lora_module_memory(const ModuleSpec& mod, const LoRAConfig& cfg) {
    long long A = static_cast<long long>(cfg.rank) * mod.d_in;
    long long B = static_cast<long long>(mod.d_out) * cfg.rank;
    double bytes_mb = (A + B) * BPE_BF16 / 1e6;
    return {mod.name, A, B, bytes_mb};
}

double full_adapter_mb(const ModelArch& arch, const LoRAConfig& cfg,
                        const std::vector<ModuleSpec>& per_layer_mods) {
    double per_layer = 0.0;
    for (const auto& mod : per_layer_mods) {
        auto bk = lora_module_memory(mod, cfg);
        per_layer += bk.bytes_mb;
    }
    return per_layer * arch.n_layers;
}

static void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "  " << title << "\n"
              << std::string(70, '=') << "\n";
}

void demo_memory_budget() {
    print_section("LoRA Adapter Memory Budget");

    // Llama 3.1 8B
    ModelArch llama8b{"Llama 3.1 8B", 32, 4096, 14336};

    // Q, K, V, O projections (all 4096×4096)
    std::vector<ModuleSpec> attn_mods = {
        {"q_proj", 4096, 4096},
        {"k_proj", 4096, 4096},
        {"v_proj", 4096, 4096},
        {"o_proj", 4096, 4096},
    };
    // + MLP
    std::vector<ModuleSpec> all_mods = attn_mods;
    all_mods.push_back({"gate_proj", 14336, 4096});
    all_mods.push_back({"up_proj",   14336, 4096});
    all_mods.push_back({"down_proj",  4096, 14336});

    struct Scenario {
        std::string label;
        LoRAConfig  cfg;
        std::vector<ModuleSpec> mods;
    };

    std::vector<Scenario> scenarios = {
        {"r=8,  q+v only",    {8,  8},  {{},   {{"q_proj",4096,4096},{"v_proj",4096,4096}}}},
        {"r=16, attn×4",      {16, 16}, {{},   attn_mods}},
        {"r=64, all×7",       {64, 64}, {{},   all_mods}},
    };
    // fix init
    for (auto& s : scenarios) s.mods = s.mods.empty() ? attn_mods : s.mods;
    scenarios[0].mods = {{"q_proj",4096,4096},{"v_proj",4096,4096}};
    scenarios[1].mods = attn_mods;
    scenarios[2].mods = all_mods;

    double base_gb = 16.0;   // Llama 3.1 8B BF16
    double avail_mb = (80.0 * 0.90 - base_gb - 1.5) * 1024.0;

    std::cout << "  " << std::left  << std::setw(22) << "Scenario"
              << std::right << std::setw(14) << "Adapter (MB)"
              << std::right << std::setw(18) << "Adapters/H100" << "\n";
    std::cout << "  " << std::string(22,'-') << " " << std::string(14,'-')
              << " " << std::string(18,'-') << "\n";

    for (const auto& s : scenarios) {
        double mb = full_adapter_mb(llama8b, s.cfg, s.mods);
        int n_adapters = static_cast<int>(avail_mb / mb);
        std::cout << "  " << std::left  << std::setw(22) << s.label
                  << std::right << std::setw(14) << std::fixed << std::setprecision(1) << mb
                  << std::right << std::setw(18) << n_adapters << "\n";
    }

    std::cout << "\n  Base model: " << base_gb << " GB  |  H100: 80 GB  "
              << "|  Available for adapters: "
              << std::fixed << std::setprecision(1) << avail_mb / 1024.0 << " GB\n";
}

void demo_lora_merge_math() {
    print_section("LoRA Merge: B·A Computation (Toy Example)");

    // Tiny example: d_out=4, d_in=4, r=2
    // A: 2×4, B: 4×2
    const int D = 4, R = 2;

    // A (r × d_in)
    double A[R][D] = {
        {0.1,  0.2, -0.1,  0.3},
        {0.0, -0.1,  0.4, -0.2},
    };
    // B (d_out × r)
    double B[D][R] = {
        { 0.5, -0.1},
        {-0.3,  0.4},
        { 0.2,  0.0},
        { 0.1, -0.2},
    };

    // Compute ΔW = B·A  (D × D result)
    double delta_W[D][D] = {};
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            for (int k = 0; k < R; ++k)
                delta_W[i][j] += B[i][k] * A[k][j];

    const double alpha = 16.0, r = R;
    double scale = alpha / r;   // 16/2 = 8

    std::cout << "  d_out=" << D << "  d_in=" << D << "  r=" << R
              << "  alpha=" << (int)alpha << "  scale=alpha/r=" << scale << "\n\n";

    std::cout << "  B · A  (ΔW before scaling):\n  ";
    for (int i = 0; i < D; ++i) {
        std::cout << (i == 0 ? "[ " : "  ");
        for (int j = 0; j < D; ++j)
            std::cout << std::setw(7) << std::fixed << std::setprecision(3) << delta_W[i][j];
        std::cout << (i == D-1 ? " ]" : "") << "\n  ";
    }

    std::cout << "\n  ΔW × scale (" << scale << "):\n  ";
    for (int i = 0; i < D; ++i) {
        std::cout << (i == 0 ? "[ " : "  ");
        for (int j = 0; j < D; ++j)
            std::cout << std::setw(7) << std::fixed << std::setprecision(3)
                      << delta_W[i][j] * scale;
        std::cout << (i == D-1 ? " ]" : "") << "\n  ";
    }

    std::cout << "\n  W_merged = W_base + ΔW × scale\n"
              << "  During inference: x @ W_merged.T  (zero overhead)\n"
              << "  vLLM injection:   x @ W_base.T + x @ A.T @ B.T × scale\n";

    // Frobenius norm of delta
    double frob = 0.0;
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            frob += (delta_W[i][j] * scale) * (delta_W[i][j] * scale);
    frob = std::sqrt(frob);
    std::cout << "\n  ||ΔW × scale||_F = " << std::fixed << std::setprecision(4)
              << frob << "  (Frobenius norm — size of the weight update)\n";
}

void demo_compression_ratio() {
    print_section("Compression Ratio: LoRA vs Full Fine-Tune");

    struct Scenario {
        std::string model;
        int n_layers, d_model, r;
    };
    std::vector<Scenario> scenarios = {
        {"Llama 3.1 8B",  32, 4096,  16},
        {"Llama 3.3 70B", 80, 8192,  16},
        {"Llama 3.3 70B", 80, 8192,  64},
    };

    std::cout << "  " << std::left  << std::setw(20) << "Model"
              << std::right << std::setw(6)  << "rank"
              << std::right << std::setw(14) << "Full FT (GB)"
              << std::right << std::setw(14) << "LoRA (MB)"
              << std::right << std::setw(12) << "Ratio" << "\n";
    std::cout << "  " << std::string(20,'-') << " " << std::string(6,'-')
              << " " << std::string(14,'-') << " " << std::string(14,'-')
              << " " << std::string(12,'-') << "\n";

    for (const auto& s : scenarios) {
        // Attention matrices only (4 per layer)
        long long full_ft_bytes = 4LL * s.n_layers * s.d_model * s.d_model * 2; // BF16
        long long lora_bytes    = 4LL * s.n_layers * 2 * s.r * s.d_model * 2;
        double full_gb = full_ft_bytes / 1e9;
        double lora_mb = lora_bytes    / 1e6;
        double ratio   = full_ft_bytes / (double)lora_bytes;
        std::cout << "  " << std::left  << std::setw(20) << s.model
                  << std::right << std::setw(6)  << s.r
                  << std::right << std::setw(14) << std::fixed << std::setprecision(1) << full_gb
                  << std::right << std::setw(14) << std::setprecision(1) << lora_mb
                  << std::right << std::setw(11) << std::setprecision(0) << ratio << "×\n";
    }
}

int main() {
    std::cout << "\nChapter 22 — LoRA Serving Demo (C++)\n";
    std::cout << std::string(70, '=') << "\n";

    demo_memory_budget();
    demo_lora_merge_math();
    demo_compression_ratio();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  Demo complete.\n";
    std::cout << std::string(70, '=') << "\n";
    return 0;
}
```

---

## §22.9  Summary

- LoRA represents fine-tuning as a low-rank delta ΔW = B·A where B ∈ ℝ^(d_out × r) and A ∈ ℝ^(r × d_in). For r = 16, this yields ~128× compression vs. storing the full weight update.
- A full LoRA adapter (attention + MLP projections, r = 16, 32 layers) is approximately 70 MB for Llama 3.1 8B — less than 0.5% of the base model size.
- An H100 80 GB can theoretically hold hundreds of adapters alongside the base model and a full KV cache. In practice, vLLM's `--max-loras` controls the hot-slot count, with a CPU warm cache (`--max-cpu-loras`) for overflow.
- vLLM injects the LoRA delta at inference time via a fused CUDA kernel, batching the base model computation across all requests while applying per-request deltas. This enables zero-reload hot-swapping.
- llama.cpp merges the adapter into base weights at load time — zero per-forward overhead but only one adapter active per context. Merging into quantized weights introduces extra quantization error; merge into FP16 first for quality-sensitive applications.
- A/B testing fine-tunes requires only a routing function that maps user IDs to `LoRARequest` objects — no infrastructure changes.

## Self-Check Questions

1. An attention weight matrix is 4096 × 4096. A LoRA adapter uses r = 8. How many parameters are in A and B combined? How many bytes at BF16?
2. Why does vLLM's batched LoRA approach maintain high GPU utilization even when requests use different adapters? What is the key efficiency gain vs. loading a separate full model per adapter?
3. When merging a LoRA adapter into a Q4_K_M quantized model in llama.cpp, why might the merged quality be lower than merging into FP16? What is the recommended mitigation?
4. You have `--max-loras 4` set. A fifth adapter is requested. Trace exactly what happens in vLLM's adapter manager.
5. Your A/B test has the candidate adapter at 10% traffic. After 1,000 requests, the stable adapter served 902 users and the candidate 98. Is this routing correct? How would you verify deterministic routing?

## Where We Go Next

Chapter 23 covers speculative decoding — the technique that breaks the one-token-per-forward-pass ceiling. A draft model proposes multiple tokens in parallel; the target model verifies all of them in a single pass. We derive the exact speedup formula, work through the acceptance rate calculation, and explore vLLM's speculative decoding configuration alongside llama.cpp's `--draft-max` path.


---

## Chapter Summary

- **LoRA fundamentals**: low-rank adapters add ΔW = BA to frozen base weights, where B ∈ ℝ^{d×r} and A ∈ ℝ^{r×k} with r ≪ min(d,k); parameter count is 2rdk vs dk for full fine-tuning.
- **Serving challenge**: serving 50 LoRA adapters naively requires 50 model replicas; LoRA serving hot-swaps adapters on a single base model.
- **vLLM LoRA support**: `--enable-lora --max-loras N --lora-modules` loads up to N adapters into GPU memory simultaneously; request routing selects the correct adapter per request.
- **Adapter memory budget**: rank-16 LoRA on a LLaMA-3 8B model ≈ 70 MB per adapter; 50 adapters = 3.5 GB total, a small fraction of the 16 GB base model.
- **Dynamic loading**: adapters not in the active set are evicted from GPU to CPU; loading time is proportional to adapter size and PCIe bandwidth.
- **Batching across adapters**: vLLM can batch requests from different adapters in the same forward pass using a segment-padded approach; throughput is lower than single-adapter batching due to the segmentation overhead.
- **llama.cpp adapter support**: `--lora PATH --lora-scaled PATH α` loads a GGUF-format LoRA; only one adapter at a time is supported in the base llama.cpp server.


---

## Worked Solutions

### Question 1
**Setup:** Attention weight matrix = 4096 x 4096, LoRA rank r=8, BF16.

**Step 1 — Parameters in A and B:**
- A has shape (4096, r) = (4096, 8) -> 32,768 parameters
- B has shape (r, 4096) = (8, 4096) -> 32,768 parameters
- Total: **65,536 parameters**

**Step 2 — Bytes at BF16 (2 bytes/param):**
```
65,536 x 2 = 131,072 bytes = 128 KB
```

The full weight matrix W is 4096 x 4096 x 2 bytes = 32 MB. The LoRA adapter is 128 KB -- a 256x compression ratio. This is why LoRA adapters can be served with negligible memory overhead per adapter.

---

### Question 2
**Why vLLM's batched LoRA maintains high GPU utilization across different adapters:**

Without batched LoRA, each adapter would require loading separate model weights into VRAM -- effectively serving N separate models. vLLM's key efficiency: the **base model weights are shared across all adapters.** The base model occupies ~140 GB HBM; each adapter adds only ~128 KB per attention layer. All adapters run the same base model forward pass, with the LoRA delta B*A*(alpha/r) computed as a small additive term.

vLLM packs requests for different adapters into the **same batch**. The forward pass runs once over all requests using the shared base model. For each linear layer, adapter-specific LoRA deltas are applied using a batched GEMM (Punica or SGMV kernels) that dispatches per-sequence adapter indices. The GPU sees a large, efficient matrix multiply regardless of which adapter each request uses -- no adapter-switching idle time.

---

### Question 3
**Merging LoRA into Q4_K_M -- why quality may be lower than FP16 merge:**

The Q4_K_M format quantizes weights to 4-bit using k-means. When merging:
```
W_merged = W_quantized_base + B*A*(alpha/r)
```
The LoRA delta is added to a lossy approximation of the original weights. After merging, W_merged must be re-quantized to Q4_K_M. This second quantization uses cluster centers optimized for W_base -- suboptimal for W_merged with different per-channel statistics.

**Recommended mitigation:**
1. Dequantize base model to FP16.
2. Merge: W_merged = W_FP16_base + B*A*(alpha/r).
3. Re-quantize W_merged to Q4_K_M from scratch, letting the quantizer optimize cluster centers for the merged weights.

---

### Question 4
**`--max-loras 4` and a fifth adapter is requested -- trace:**

1. Request arrives with adapter_id=5. Manager checks in-memory registry: 4 adapters loaded, cache full.
2. LRU eviction: adapter with oldest last-use timestamp is evicted from GPU VRAM (A and B matrices freed; base model unchanged).
3. Adapter 5's weight file is loaded from disk into CPU memory and transferred to GPU VRAM (~50-200 ms I/O cost).
4. Request is admitted to the scheduler queue and processed normally.

For production: set --max-loras to the expected concurrently-needed adapter count, not total adapter count, to avoid I/O eviction overhead.

---

### Question 5
**A/B test: stable 90% / candidate 10%. After 1,000 requests: stable=902, candidate=98.**

**Is routing correct?** Expected = 100 candidate. Observed = 98. Deviation = 2% -- within normal statistical noise. Routing appears correct.

**Verify deterministic routing:**
Use hash-based routing: route = "candidate" if hash(user_id) % 100 < 10 else "stable". This ensures:

1. The same user always sees the same adapter (no A/B contamination across sessions).
2. The 10% split is deterministic, not probabilistic.

Verification: sample 10,000 synthetic user IDs, confirm 900-1100 (9-11%) route to candidate. Log adapter_id per request and alert if rolling ratio deviates >5% from target in production.

