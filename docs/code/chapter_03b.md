# Code 3.5 — The Transformer Block

## Python — transformer_block_demo.py

```python
#!/usr/bin/env python3
"""
transformer_block_demo.py — Chapter 3.5 Companion Code

Implements a complete transformer block from scratch:
  RMSNorm, SwiGLU FFN, stub attention, residual stream.
Computes parameter counts, KV cache sizes, and FLOPs
for LLaMA-3 model families.

Run: python3 transformer_block_demo.py
All assertions self-verify the worked examples in Chapter 3.5.
"""
import math

class ModelConfig:
    def __init__(self, name, hidden_size, intermediate_size, num_layers,
                 num_attention_heads, num_kv_heads, vocab_size):
        self.name = name
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.vocab_size = vocab_size
        self.head_dim = hidden_size // num_attention_heads

# LLaMA-3 family configs
LLAMA3_1B  = ModelConfig("LLaMA-3-1B",  2048,  8192,  16, 32, 8,  32000)
LLAMA3_8B  = ModelConfig("LLaMA-3-8B",  4096, 14336,  32, 32, 8,  128256)
LLAMA3_70B = ModelConfig("LLaMA-3-70B", 8192, 28672,  80, 64, 8,  128256)

class RMSNorm:
    """Root Mean Square Layer normalization (no mean subtraction)."""
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

    def params(self) -> int:
        return self.hidden_size  # just the scale vector γ

    def forward(self, x: list[float]) -> list[float]:
        rms = math.sqrt(sum(v*v for v in x) / len(x) + 1e-6)
        return [v / rms for v in x]  # γ=1 for simplicity

class SwiGLU_FFN:
    """SwiGLU feed-forward: output = (SiLU(W1·x) ⊙ W3·x) · W2"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.d = hidden_size
        self.di = intermediate_size

    def params(self) -> int:
        return 3 * self.d * self.di  # W1, W2, W3

    def forward(self, x: list[float]) -> list[float]:
        # Toy single-element demonstration
        gate = math.tanh(x[0])        # approximate SiLU
        hidden = gate * x[0]          # gated activation
        return [hidden * 0.1]         # W2 projection (stub)

class TransformerBlock:
    """One complete transformer block: norm → attn → residual → norm → ffn → residual."""
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.ffn   = SwiGLU_FFN(cfg.hidden_size, cfg.intermediate_size)

    def params(self) -> int:
        d, h = self.cfg.hidden_size, self.cfg.head_dim
        nq = self.cfg.num_attention_heads
        nkv = self.cfg.num_kv_heads
        attn = (nq + 2*nkv) * h * d + d * d   # Q,K,V + out proj
        return attn + self.ffn.params() + self.norm1.params() + self.norm2.params()

def model_memory_gb(cfg: ModelConfig, dtype_bytes: int = 2) -> float:
    per_block = TransformerBlock(cfg).params()
    total = per_block * cfg.num_layers
    total += cfg.vocab_size * cfg.hidden_size * 2  # embed + unembed
    return total * dtype_bytes / 1e9

def kv_cache_gb(cfg: ModelConfig, seq_len: int, batch: int,
                dtype_bytes: int = 2) -> float:
    bytes_per_token = (2 * cfg.num_kv_heads * cfg.head_dim *
                       cfg.num_layers * dtype_bytes)
    return bytes_per_token * seq_len * batch / 1e9

def flops_per_token(cfg: ModelConfig) -> float:
    d = cfg.hidden_size
    # Attention: 4d² (Q,K,V,O projections) + 2d² (softmax attention ops ≈ d)
    attn = 4 * d * d
    # FFN: 3 × d × di (three matmuls in SwiGLU)
    ffn = 3 * d * cfg.intermediate_size
    return 2 * cfg.num_layers * (attn + ffn)  # ×2 for multiply-add

if __name__ == "__main__":
    print("=== Transformer Block — Chapter 3.5 ===\n")
    for cfg in [LLAMA3_1B, LLAMA3_8B, LLAMA3_70B]:
        mem  = model_memory_gb(cfg)
        kv   = kv_cache_gb(cfg, seq_len=4096, batch=1)
        flps = flops_per_token(cfg)
        print(f"{cfg.name}:")
        print(f"  Model weights (BF16): {mem:.1f} GB")
        print(f"  KV cache @ 4096 tok, batch=1 (BF16): {kv:.3f} GB")
        print(f"  FLOPs/token (decode): {flps/1e9:.1f} GFLOPs")
        print()

    # Verify LLaMA-3 8B known values
    mem8b = model_memory_gb(LLAMA3_8B)
    assert 14 < mem8b < 18, f"8B weight memory out of range: {mem8b:.1f} GB"
    kv8b = kv_cache_gb(LLAMA3_8B, 4096, 1)
    assert 0.5 < kv8b < 2.0, f"8B KV cache out of range: {kv8b:.3f} GB"
    print("All assertions passed.")
```

## C++ — transformer_block_demo.cpp

```cpp
// transformer_block_demo.cpp — Chapter 3.5 Companion Code
// Compile: g++ -O2 -std=c++17 -o transformer_block_demo transformer_block_demo.cpp
// Run:     ./transformer_block_demo

#include <cmath>
#include <cassert>
#include <iostream>
#include <string>

struct ModelConfig {
    std::string name;
    int hidden_size, intermediate_size, num_layers;
    int num_attention_heads, num_kv_heads, vocab_size;
    int head_dim() const { return hidden_size / num_attention_heads; }
};

// LLaMA-3 family
constexpr ModelConfig LLAMA3_8B  = {"LLaMA-3-8B",  4096, 14336, 32, 32, 8, 128256};
constexpr ModelConfig LLAMA3_70B = {"LLaMA-3-70B", 8192, 28672, 80, 64, 8, 128256};

long long block_params(const ModelConfig& c) {
    int d = c.hidden_size, hd = c.head_dim();
    long long attn = (long long)(c.num_attention_heads + 2*c.num_kv_heads) * hd * d + (long long)d*d;
    long long ffn  = 3LL * d * c.intermediate_size;
    long long norm = 2LL * d;
    return attn + ffn + norm;
}

double model_memory_gb(const ModelConfig& c, int dtype_bytes = 2) {
    long long total = block_params(c) * c.num_layers
                    + 2LL * c.vocab_size * c.hidden_size;
    return total * dtype_bytes / 1e9;
}

double kv_cache_gb(const ModelConfig& c, int seq_len, int batch, int dtype_bytes = 2) {
    long long bpt = 2LL * c.num_kv_heads * c.head_dim() * c.num_layers * dtype_bytes;
    return bpt * seq_len * batch / 1e9;
}

double flops_per_token(const ModelConfig& c) {
    long long attn = 4LL * c.hidden_size * c.hidden_size;
    long long ffn  = 3LL * c.hidden_size * c.intermediate_size;
    return 2.0 * c.num_layers * (attn + ffn);
}

int main() {
    std::cout << "=== Transformer Block Demo (C++) ===\n\n";
    for (const auto& cfg : {LLAMA3_8B, LLAMA3_70B}) {
        double mem  = model_memory_gb(cfg);
        double kv   = kv_cache_gb(cfg, 4096, 1);
        double flps = flops_per_token(cfg);
        std::cout << cfg.name << ":\n";
        std::cout << "  Model weights (BF16): " << mem  << " GB\n";
        std::cout << "  KV cache @4096 tok:   " << kv   << " GB\n";
        std::cout << "  FLOPs/token:          " << flps/1e9 << " GFLOPs\n\n";
    }

    double mem8b = model_memory_gb(LLAMA3_8B);
    assert(mem8b > 14.0 && mem8b < 18.0);
    double kv8b = kv_cache_gb(LLAMA3_8B, 4096, 1);
    assert(kv8b > 0.5 && kv8b < 2.0);
    std::cout << "All assertions passed.\n";
    return 0;
}
```
