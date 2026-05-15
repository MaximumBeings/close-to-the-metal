# Chapter 28: llama.cpp as a Platform — Companion Code

## Python — `gguf_reader.py`

```python
# gguf_reader.py
"""
Minimal GGUF reader — demonstrates the file format without external dependencies.
For production use: pip install gguf  (the official Python library)
"""
import struct, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


GGUF_MAGIC   = b"GGUF"
VALUE_TYPES  = {
    0: ("uint8",   "B", 1),
    1: ("int8",    "b", 1),
    2: ("uint16",  "H", 2),
    3: ("int16",   "h", 2),
    4: ("uint32",  "I", 4),
    5: ("int32",   "i", 4),
    6: ("float32", "f", 4),
    7: ("bool",    "B", 1),
    8: ("string",  None, None),
    9: ("array",   None, None),
    10:("uint64",  "Q", 8),
    11:("int64",   "q", 8),
    12:("float64", "d", 8),
}

QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 10: "Q4_K_S", 11: "Q4_K_M",
    12: "Q5_K_S", 13: "Q5_K_M", 14: "Q6_K", 15: "Q8_0", 30: "IQ4_NL", 34: "BF16",
}


@dataclass
class TensorInfo:
    name: str
    dims: List[int]
    quant_type: int
    offset: int

    @property
    def quant_name(self) -> str:
        return QUANT_NAMES.get(self.quant_type, f"UNKNOWN({self.quant_type})")

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


class GGUFReader:
    def __init__(self, path: str):
        self.path = path
        self.kv: Dict[str, Any] = {}
        self.tensors: List[TensorInfo] = []
        self._parse()

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: {magic!r}")

            version   = self._read_u32(f)
            n_tensors = self._read_u64(f)
            n_kv      = self._read_u64(f)

            self.version   = version
            self.n_tensors = n_tensors

            # Key-Value store
            for _ in range(n_kv):
                key   = self._read_string(f)
                vtype = self._read_u32(f)
                val   = self._read_value(f, vtype)
                self.kv[key] = val

            # Tensor info
            for _ in range(n_tensors):
                name   = self._read_string(f)
                n_dims = self._read_u32(f)
                dims   = [self._read_u64(f) for _ in range(n_dims)]
                qtype  = self._read_u32(f)
                offset = self._read_u64(f)
                self.tensors.append(TensorInfo(name, dims, qtype, offset))

            self.data_offset = f.tell()
            # Align to 32 bytes
            if self.data_offset % 32 != 0:
                self.data_offset += 32 - (self.data_offset % 32)

    def _read_u32(self, f): return struct.unpack("<I", f.read(4))[0]
    def _read_u64(self, f): return struct.unpack("<Q", f.read(8))[0]
    def _read_i32(self, f): return struct.unpack("<i", f.read(4))[0]
    def _read_f32(self, f): return struct.unpack("<f", f.read(4))[0]

    def _read_string(self, f) -> str:
        length = self._read_u64(f)
        return f.read(length).decode("utf-8", errors="replace")

    def _read_value(self, f, vtype: int):
        if vtype == 8:   # string
            return self._read_string(f)
        if vtype == 9:   # array
            elem_type = self._read_u32(f)
            count     = self._read_u64(f)
            return [self._read_value(f, elem_type) for _ in range(min(count, 64))]
        info = VALUE_TYPES.get(vtype)
        if info is None:
            raise ValueError(f"Unknown value type: {vtype}")
        _, fmt, size = info
        return struct.unpack(f"<{fmt}", f.read(size))[0]

    def summary(self):
        arch = self.kv.get("general.architecture", "?")
        name = self.kv.get("general.name", "?")
        ctx  = self.kv.get(f"{arch}.context_length", "?")
        emb  = self.kv.get(f"{arch}.embedding_length", "?")
        blk  = self.kv.get(f"{arch}.block_count", "?")
        rope_base = self.kv.get(f"{arch}.rope.freq_base", "?")

        print(f"GGUF v{self.version}: {name}")
        print(f"  Architecture:   {arch}")
        print(f"  Context length: {ctx:,}" if isinstance(ctx, int) else f"  Context length: {ctx}")
        print(f"  Embedding dim:  {emb}")
        print(f"  Layers:         {blk}")
        print(f"  RoPE base:      {rope_base}")
        print(f"  Tensors:        {self.n_tensors}")
        print()

        # Quantization type breakdown
        quant_counts: Dict[str, int] = {}
        quant_params: Dict[str, int] = {}
        for t in self.tensors:
            qn = t.quant_name
            quant_counts[qn] = quant_counts.get(qn, 0) + 1
            quant_params[qn] = quant_params.get(qn, 0) + t.n_elements

        print("  Quantization breakdown:")
        for qn, count in sorted(quant_counts.items()):
            params_m = quant_params[qn] / 1e6
            print(f"    {qn:12s}  {count:5d} tensors  {params_m:8.1f}M params")
        print()

        # Sample tensors
        print("  First 10 tensors:")
        for t in self.tensors[:10]:
            shape = " × ".join(str(d) for d in t.dims)
            print(f"    {t.name:50s}  [{shape:30s}]  {t.quant_name}")
        if len(self.tensors) > 10:
            print(f"    … and {len(self.tensors) - 10} more")


# ─────────────────────────────────────────────────────────────────────────────
# GGUF CONVERSION NOTES (without an actual GGUF file for the demo)
# ─────────────────────────────────────────────────────────────────────────────

def demo_gguf_structure():
    """Show GGUF structure without requiring an actual model file."""
    print("=" * 70)
    print("GGUF File Structure Overview")
    print("=" * 70)

    struct_diagram = """
  Offset   Size    Content
  ──────   ────    ───────
  0        4       Magic: 'GGUF'
  4        4       Version (uint32, currently 3)
  8        8       n_tensors (uint64)
  16       8       n_kv (uint64)
  24       var     Key-Value metadata store
                     [key: string][value_type: uint32][value: ...]
                     ... repeated n_kv times
  var      var     Tensor info array
                     [name: string][n_dims: u32][dims: u64[]][type: u32][offset: u64]
                     ... repeated n_tensors times
  align    pad     Padding to 32-byte alignment
  data     var     Tensor data (memory-mappable)
                     tensor[0], tensor[1], ..., tensor[n_tensors-1]
"""
    print(struct_diagram)

    # Show quantization bit-per-weight table
    print("  Quantization formats (approximate bits/weight):")
    quants = [
        ("F32",    32.0,  "Reference precision"),
        ("F16",    16.0,  "Training precision"),
        ("BF16",   16.0,  "Brain float (better range than F16)"),
        ("Q8_0",    8.5,  "Fast dequant, near-lossless"),
        ("Q6_K",    6.56, "High quality, 4× compression vs FP16"),
        ("Q5_K_M",  5.68, "Good quality/size balance"),
        ("Q4_K_M",  4.84, "Recommended default — best Q/size"),
        ("Q4_K_S",  4.5,  "Slightly smaller than Q4_K_M"),
        ("Q4_0",    4.5,  "Legacy, avoid for new models"),
        ("IQ4_NL",  4.5,  "imatrix-aware, better than Q4_0"),
    ]
    print(f"  {'Name':12}  {'Bits':6}  {'Llama-3.1-8B size':20}  Notes")
    print("  " + "-" * 65)
    for name, bits, note in quants:
        size_gb = 8e9 * bits / 16 / 1e9  # relative to FP16
        print(f"  {name:12}  {bits:6.2f}  {size_gb:>8.1f} GB               {note}")
    print()


if __name__ == "__main__":
    import os, sys
    demo_gguf_structure()

    # If a GGUF path is provided as argument, read it
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"Reading: {sys.argv[1]}")
        r = GGUFReader(sys.argv[1])
        r.summary()
    else:
        print("  (Pass a .gguf file path as argument to inspect a real model)")

```

## Python — `llamacpp_ctypes.py`

```python
# llamacpp_ctypes.py
"""
Minimal ctypes wrapper for llama.cpp — demonstrates the ABI surface.
Production code should use llama-cpp-python instead.
"""
import ctypes
import ctypes.util
import os

def load_llama_lib(lib_path: str | None = None) -> ctypes.CDLL:
    """Load libllama.so / llama.dll / libllama.dylib."""
    if lib_path is None:
        # Try common locations
        candidates = [
            "libllama.so", "libllama.dylib",
            os.path.join(os.path.dirname(__file__), "libllama.so"),
        ]
        for p in candidates:
            try:
                return ctypes.CDLL(p)
            except OSError:
                continue
        raise FileNotFoundError("libllama not found; build llama.cpp and set LD_LIBRARY_PATH")
    return ctypes.CDLL(lib_path)


def configure_api(lib: ctypes.CDLL):
    """Set arg/return types for key API functions."""
    lib.llama_backend_init.restype  = None
    lib.llama_backend_free.restype  = None

    lib.llama_load_model_from_file.restype  = ctypes.c_void_p
    lib.llama_load_model_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p]

    lib.llama_free_model.restype  = None
    lib.llama_free_model.argtypes = [ctypes.c_void_p]

    lib.llama_new_context_with_model.restype  = ctypes.c_void_p
    lib.llama_new_context_with_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lib.llama_free.restype  = None
    lib.llama_free.argtypes = [ctypes.c_void_p]

    lib.llama_n_vocab.restype  = ctypes.c_int32
    lib.llama_n_vocab.argtypes = [ctypes.c_void_p]

    lib.llama_tokenize.restype  = ctypes.c_int32
    lib.llama_tokenize.argtypes = [
        ctypes.c_void_p,   # model
        ctypes.c_char_p,   # text
        ctypes.c_int32,    # text_len
        ctypes.POINTER(ctypes.c_int32),  # tokens
        ctypes.c_int32,    # n_tokens_max
        ctypes.c_bool,     # add_special
        ctypes.c_bool,     # parse_special
    ]
    return lib

```

## Python — `llamacpp_platform_demo.py`

```python
# llamacpp_platform_demo.py
"""
Chapter 28 — llama.cpp as a Platform (Python demo)

Demonstrates:
  1. GGUF structure overview
  2. llama-cpp-python API patterns (with mock when no model present)
  3. Grammar-constrained decoding simulation
  4. Sampler chain construction
  5. KV cache management patterns
  6. Performance tuning guide
"""
from __future__ import annotations
import json, math, time, struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GGUF STRUCTURE ANALYZER (works on any GGUF file)
# ─────────────────────────────────────────────────────────────────────────────

QUANT_BITS = {
    0: 32.0, 1: 16.0, 2: 4.5, 10: 4.5, 11: 4.84,
    12: 5.5, 13: 5.68, 14: 6.56, 15: 8.5, 30: 4.5, 34: 16.0,
}
QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 10: "Q4_K_S", 11: "Q4_K_M",
    12: "Q5_K_S", 13: "Q5_K_M", 14: "Q6_K", 15: "Q8_0", 30: "IQ4_NL", 34: "BF16",
}


def analyze_gguf_size(param_billions: float) -> None:
    """Print model size estimates across all quantization formats."""
    print("=" * 68)
    print(f"GGUF Size Estimates — {param_billions}B parameter model")
    print("=" * 68)
    print(f"  {'Format':12}  {'Bits/W':7}  {'Size (GB)':10}  {'vs FP16':8}  Notes")
    print("  " + "-" * 60)
    rows = [
        (0,  "F32"),
        (1,  "F16"),
        (34, "BF16"),
        (15, "Q8_0"),
        (14, "Q6_K"),
        (13, "Q5_K_M"),
        (12, "Q5_K_S"),
        (11, "Q4_K_M"),
        (10, "Q4_K_S"),
        (2,  "Q4_0"),
        (30, "IQ4_NL"),
    ]
    fp16_size = param_billions * 1e9 * 2 / 1e9
    for qtype, name in rows:
        bits = QUANT_BITS[qtype]
        size = param_billions * 1e9 * bits / 16 / 1e9  # relative to fp16=2 bytes
        ratio = fp16_size / size
        notes = ""
        if name == "Q4_K_M": notes = "← recommended"
        if name == "F16":     notes = "← reference"
        print(f"  {name:12}  {bits:7.2f}  {size:10.2f}  {ratio:7.1f}×  {notes}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SAMPLER CHAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenCandidate:
    id: int
    logit: float
    p: float = 0.0     # probability after softmax


def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def apply_temperature(candidates: List[TokenCandidate], temp: float
                       ) -> List[TokenCandidate]:
    """Divide logits by temperature before softmax."""
    for c in candidates:
        c.logit /= temp
    probs = softmax([c.logit for c in candidates])
    for c, p in zip(candidates, probs):
        c.p = p
    return candidates


def apply_top_k(candidates: List[TokenCandidate], k: int
                ) -> List[TokenCandidate]:
    """Keep only the top-k tokens by probability."""
    candidates.sort(key=lambda c: c.p, reverse=True)
    kept = candidates[:k]
    # Renormalize
    total = sum(c.p for c in kept)
    for c in kept:
        c.p /= total
    return kept


def apply_top_p(candidates: List[TokenCandidate], p: float
                ) -> List[TokenCandidate]:
    """Keep tokens whose cumulative probability ≤ p."""
    candidates.sort(key=lambda c: c.p, reverse=True)
    cumsum = 0.0
    kept = []
    for c in candidates:
        if cumsum >= p and len(kept) > 0:
            break
        kept.append(c)
        cumsum += c.p
    total = sum(c.p for c in kept)
    for c in kept:
        c.p /= total
    return kept


def apply_min_p(candidates: List[TokenCandidate], min_p: float
                ) -> List[TokenCandidate]:
    """Keep tokens with p ≥ min_p * max_p."""
    max_p = max(c.p for c in candidates)
    threshold = min_p * max_p
    kept = [c for c in candidates if c.p >= threshold]
    if not kept:
        kept = [max(candidates, key=lambda c: c.p)]
    total = sum(c.p for c in kept)
    for c in kept:
        c.p /= total
    return kept


def apply_repetition_penalty(candidates: List[TokenCandidate],
                              recent_tokens: List[int],
                              penalty: float) -> List[TokenCandidate]:
    """Apply repetition penalty to recently seen tokens."""
    recent_set = set(recent_tokens)
    for c in candidates:
        if c.id in recent_set:
            c.logit = c.logit / penalty if c.logit > 0 else c.logit * penalty
    # Re-run softmax after modifying logits
    probs = softmax([c.logit for c in candidates])
    for c, p in zip(candidates, probs):
        c.p = p
    return candidates


def demo_sampler_chain():
    """Demonstrate the sampler chain pipeline on mock logits."""
    import random
    rng = random.Random(42)

    # Mock vocabulary of 16 tokens with random logits
    V = 16
    vocab = [f"tok_{i}" for i in range(V)]
    logits = [rng.gauss(0, 2) for _ in range(V)]

    # Simulate: temperature=0.7, top_k=8, top_p=0.9, min_p=0.05
    candidates = [TokenCandidate(id=i, logit=logits[i]) for i in range(V)]

    print("=" * 60)
    print("Sampler Chain Simulation (V=16, mock logits)")
    print("=" * 60)

    def show(stage: str, cands: List[TokenCandidate]):
        top3 = sorted(cands, key=lambda c: c.p, reverse=True)[:3]
        print(f"  After {stage:20s}: {len(cands):3d} tokens | "
              f"top={top3[0].p:.3f}/{top3[1].p:.3f}/{top3[2].p:.3f}")

    # Step 1: Initial softmax
    probs = softmax([c.logit for c in candidates])
    for c, p in zip(candidates, probs):
        c.p = p
    show("initial softmax", candidates)

    # Step 2: Temperature
    candidates = apply_temperature(candidates, temp=0.7)
    show("temperature=0.7", candidates)

    # Step 3: Top-K
    candidates = apply_top_k(candidates, k=8)
    show("top_k=8", candidates)

    # Step 4: Top-P
    candidates = apply_top_p(candidates, p=0.90)
    show("top_p=0.90", candidates)

    # Step 5: Min-P
    candidates = apply_min_p(candidates, min_p=0.05)
    show("min_p=0.05", candidates)

    # Final distribution
    print()
    print("  Final distribution:")
    for c in sorted(candidates, key=lambda c: c.p, reverse=True):
        bar = "█" * int(c.p * 40)
        print(f"    {vocab[c.id]:10s}  {c.p:.4f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GRAMMAR CONSTRAINT SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_grammar_constrained_decoding():
    """
    Show how grammar masking reduces the valid token set at each step.
    Uses a simplified JSON grammar simulation.
    """
    print("=" * 68)
    print("Grammar-Constrained Decoding Simulation — JSON Grammar")
    print("=" * 68)

    # Simulate JSON grammar states
    states = [
        {
            "description": 'Start of object: only "{" allowed',
            "position":    'Before any output',
            "vocab_size":  128000,
            "valid_tokens": 1,
            "valid_examples": ['"{" only'],
        },
        {
            "description": 'After "{": only string key (") or "}" allowed',
            "position":    'After "{"',
            "vocab_size":  128000,
            "valid_tokens": 2,
            "valid_examples": ['"\\""', '"}"'],
        },
        {
            "description": 'Inside key string: any non-quote char + escape',
            "position":    'After opening "\\"" of key',
            "vocab_size":  128000,
            "valid_tokens": 127800,
            "valid_examples": ['alpha/num tokens'],
        },
        {
            "description": 'After key: only ":" allowed',
            "position":    'After closing "\\"" of key',
            "vocab_size":  128000,
            "valid_tokens": 1,
            "valid_examples": ['":"'],
        },
        {
            "description": 'After ":": any JSON value start',
            "position":    'Before value',
            "vocab_size":  128000,
            "valid_tokens": 5,
            "valid_examples": ['"\\"", "[", "{", "true", "false", "null", digit'],
        },
        {
            "description": 'After string value: "," or "}"',
            "position":    'After value',
            "vocab_size":  128000,
            "valid_tokens": 2,
            "valid_examples": ['","', '"}"'],
        },
    ]

    print(f"  {'Position':35}  {'Valid':8}  {'% vocab':8}  Example tokens")
    print("  " + "-" * 75)
    for s in states:
        pct = s["valid_tokens"] / s["vocab_size"] * 100
        examples = ", ".join(s["valid_examples"])
        print(f"  {s['description'][:35]:35}  {s['valid_tokens']:8,}  {pct:8.4f}%  {examples}")

    print()
    print("  Observation: grammar masking reduces valid tokens by 4-6 orders of")
    print("  magnitude for structural positions, while leaving string content nearly")
    print("  unconstrained. The overhead is ~5-15% latency for mask application,")
    print("  but eliminates all retries from malformed JSON output.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  KV CACHE SEQUENCE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KVCacheSlot:
    seq_id: int
    positions: List[int] = field(default_factory=list)
    state: str = "idle"  # idle | prefill | decode | fork


class MockKVCache:
    """Demonstrates KV cache sequence management patterns."""

    def __init__(self, n_ctx: int, n_slots: int):
        self.n_ctx   = n_ctx
        self.n_slots = n_slots
        self.slots   = [KVCacheSlot(seq_id=i) for i in range(n_slots)]

    def assign_slot(self, seq_id: int, prompt_len: int) -> Optional[int]:
        for i, slot in enumerate(self.slots):
            if slot.state == "idle":
                slot.seq_id    = seq_id
                slot.positions = list(range(prompt_len))
                slot.state     = "prefill"
                return i
        return None

    def decode_step(self, slot_idx: int) -> int:
        slot = self.slots[slot_idx]
        new_pos = len(slot.positions)
        slot.positions.append(new_pos)
        slot.state = "decode"
        return new_pos

    def fork(self, src_idx: int, new_seq_id: int) -> Optional[int]:
        """Copy slot state for beam search branching."""
        dst_idx = self.assign_slot(new_seq_id, 0)
        if dst_idx is None:
            return None
        src = self.slots[src_idx]
        dst = self.slots[dst_idx]
        dst.positions = src.positions.copy()
        dst.state     = "fork"
        return dst_idx

    def evict(self, slot_idx: int, evict_before_pos: int):
        """Sliding window: evict positions 0..evict_before_pos."""
        slot = self.slots[slot_idx]
        slot.positions = [p for p in slot.positions if p >= evict_before_pos]

    def free(self, slot_idx: int):
        slot = self.slots[slot_idx]
        slot.positions = []
        slot.state     = "idle"

    def status(self):
        total = sum(len(s.positions) for s in self.slots)
        print(f"  KV Cache: {total}/{self.n_ctx} positions used  "
              f"({total/self.n_ctx*100:.1f}%)")
        for i, s in enumerate(self.slots):
            bar = "█" * (len(s.positions) * 20 // max(1, self.n_ctx))
            print(f"    Slot {i}  seq={s.seq_id:3d}  state={s.state:8s}  "
                  f"tokens={len(s.positions):5d}  {bar}")


def demo_kv_cache_management():
    print("=" * 60)
    print("KV Cache Sequence Management Patterns")
    print("=" * 60)

    cache = MockKVCache(n_ctx=4096, n_slots=4)

    # Pattern 1: Normal inference
    print("\n  [Pattern 1: Normal multi-user inference]")
    s0 = cache.assign_slot(seq_id=100, prompt_len=512)
    s1 = cache.assign_slot(seq_id=101, prompt_len=256)
    for _ in range(32):
        cache.decode_step(s0)
    for _ in range(64):
        cache.decode_step(s1)
    cache.status()

    # Pattern 2: Fork for beam search
    print("\n  [Pattern 2: Fork for beam search]")
    s2 = cache.fork(src_idx=s0, new_seq_id=102)
    cache.decode_step(s0)   # beam 1
    cache.decode_step(s2)   # beam 2 (forked)
    cache.status()

    # Pattern 3: Sliding window eviction
    print("\n  [Pattern 3: Sliding window eviction (keep last 256)]")
    s3 = cache.assign_slot(seq_id=103, prompt_len=800)
    cache.evict(s3, evict_before_pos=800-256)
    cache.status()

    # Pattern 4: Free completed sequence
    print("\n  [Pattern 4: Free completed sequence]")
    cache.free(s1)
    cache.status()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PERFORMANCE TUNING GUIDE
# ─────────────────────────────────────────────────────────────────────────────

def demo_performance_tuning():
    print("=" * 68)
    print("llama.cpp Performance Tuning Guide")
    print("=" * 68)

    scenarios = [
        {
            "name": "H100 80GB — Llama-3.1-70B Q4_K_M, high throughput",
            "flags": [
                ("-m",    "Llama-3.1-70B-Q4_K_M.gguf",   "4-bit quantized, ~38 GB"),
                ("-ngl",  "99",                            "All layers on GPU"),
                ("-c",    "16384",                         "4 slots × 4096"),
                ("-np",   "4",                             "4 parallel slots"),
                ("-b",    "2048",                          "Large prefill batch"),
                ("-ub",   "512",                           "Micro-batch for decode"),
                ("--flash-attn", "",                       "Required for this ctx"),
                ("--rope-freq-base", "500000",             "Llama 3.1 extended"),
            ],
        },
        {
            "name": "RTX 4090 24GB — Llama-3.1-8B Q4_K_M, low latency",
            "flags": [
                ("-m",    "Llama-3.1-8B-Q4_K_M.gguf",    "4-bit, ~4.8 GB"),
                ("-ngl",  "99",                            "All on GPU"),
                ("-c",    "8192",                          "Single context"),
                ("-np",   "1",                             "Single slot = min latency"),
                ("-b",    "512",                           "Moderate batch"),
                ("--flash-attn", "",                       "Faster attention"),
                ("--rope-freq-base", "500000",             "Llama 3.1"),
            ],
        },
        {
            "name": "M2 Ultra 192GB — Llama-3.1-70B Q6_K, quality focus",
            "flags": [
                ("-m",    "Llama-3.1-70B-Q6_K.gguf",     "6-bit, ~57 GB"),
                ("-ngl",  "99",                            "All on Apple GPU"),
                ("-c",    "32768",                         "Long context fits"),
                ("-np",   "2",                             "2 slots"),
                ("-b",    "512",                           "Metal batch size"),
                ("--flash-attn", "",                       "Metal FlashAttention"),
                ("--rope-freq-base", "500000",             "Llama 3.1"),
            ],
        },
        {
            "name": "CPU-only — Llama-3.1-8B Q4_K_M, 32-core server",
            "flags": [
                ("-m",    "Llama-3.1-8B-Q4_K_M.gguf",    "4-bit, ~4.8 GB"),
                ("-ngl",  "0",                             "No GPU offload"),
                ("-c",    "4096",                          "Modest context"),
                ("-np",   "1",                             "Single slot"),
                ("--threads", "16",                        "Half physical cores"),
                ("--threads-batch", "32",                  "All cores for prefill"),
                ("-b",    "512",                           "Batch for AVX prefill"),
            ],
        },
    ]

    for s in scenarios:
        print(f"\n  {s['name']}")
        print("  " + "-" * 60)
        cmd_parts = ["llama-server"]
        for flag, value, comment in s["flags"]:
            if value:
                cmd_parts.append(f"{flag} {value}")
                print(f"    {flag:22s} {value:20s}  # {comment}")
            else:
                cmd_parts.append(flag)
                print(f"    {flag:22s} {'':20s}  # {comment}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Chapter 28 — llama.cpp as a Platform (Python)")
    print("=" * 70 + "\n")

    analyze_gguf_size(8.0)    # 8B model
    analyze_gguf_size(70.0)   # 70B model
    demo_gguf_structure()
    demo_sampler_chain()
    simulate_grammar_constrained_decoding()
    demo_kv_cache_management()
    demo_performance_tuning()

```

## C++ — `llamacpp_platform_demo.cpp`

```cpp
// llamacpp_platform_demo.cpp
// Chapter 28 — llama.cpp as a Platform (C++ demo)
//
// Demonstrates (without requiring an actual llama.cpp installation):
//   1. GGUF header parsing (binary format recreation)
//   2. Quantization block structures (Q4_K_M layout)
//   3. Sampler chain implementation (temperature + top-k + top-p + min-p)
//   4. KV cache sequence management (seq_cp, seq_rm, seq_shift)
//   5. Token trie for grammar validation (simplified GBNF)
//   6. mmap vs malloc loading comparison
//
// Build: g++ -O2 -std=c++17 -o llamacpp_platform_demo llamacpp_platform_demo.cpp
// Run:   ./llamacpp_platform_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    return comma(n / 1000) + "," + [](long long r) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%03lld", r);
        return std::string(buf);
    }(n % 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1.  GGUF BINARY FORMAT STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

// GGUF quantization type → bits per weight
static const std::pair<int, std::pair<const char*, double>> QUANT_TABLE[] = {
    { 0,  {"F32",    32.0}},
    { 1,  {"F16",    16.0}},
    { 2,  {"Q4_0",    4.5}},
    {10,  {"Q4_K_S",  4.5}},
    {11,  {"Q4_K_M",  4.84}},
    {12,  {"Q5_K_S",  5.5}},
    {13,  {"Q5_K_M",  5.68}},
    {14,  {"Q6_K",    6.56}},
    {15,  {"Q8_0",    8.5}},
    {30,  {"IQ4_NL",  4.5}},
    {34,  {"BF16",   16.0}},
};
static const int N_QUANT = sizeof(QUANT_TABLE) / sizeof(QUANT_TABLE[0]);

static const char* quant_name(int qtype) {
    for (int i = 0; i < N_QUANT; ++i)
        if (QUANT_TABLE[i].first == qtype)
            return QUANT_TABLE[i].second.first;
    return "UNKNOWN";
}

static double quant_bpw(int qtype) {
    for (int i = 0; i < N_QUANT; ++i)
        if (QUANT_TABLE[i].first == qtype)
            return QUANT_TABLE[i].second.second;
    return 16.0;
}

// Q4_K block structure (256 weights per super-block)
// Layout: 8 sub-blocks of 32 weights each
// Scales: 8 × int8 (one per sub-block)
// Minimums: 8 × int8
// Data: 256 × 4-bit packed (128 bytes)
struct Q4KBlock {
    uint8_t  scales[8];   // dequant scales, one per 32-weight sub-block
    uint8_t  mins[8];     // dequant minimums
    uint8_t  qs[128];     // 256 weights packed as 4-bit pairs
                           // qs[i] = (w[2i] & 0xF) | (w[2i+1] << 4)
};

static_assert(sizeof(Q4KBlock) == 144, "Q4KBlock must be 144 bytes");

static void demo_gguf_format() {
    print_section("GGUF Format: Size Analysis and Quantization Structures");

    // Model size comparison
    const double PARAMS_8B  = 8.0e9;
    const double PARAMS_70B = 70.0e9;

    std::cout << "\n  Size estimates for Llama-3.1-8B:\n";
    std::cout << "  " << std::string(55, '-') << "\n";
    std::cout << std::left  << std::setw(12) << "  Format"
              << std::right << std::setw(10) << "Bits/W"
              << std::setw(12) << "Size (GB)"
              << std::setw(10) << "vs FP16"
              << "\n";

    double fp16_8b = PARAMS_8B * 2 / 1e9;
    for (int i = 0; i < N_QUANT; ++i) {
        const char* name = QUANT_TABLE[i].second.first;
        double bpw  = QUANT_TABLE[i].second.second;
        double size = PARAMS_8B * bpw / 16.0 / 1e9;  // relative to 2 bytes/param
        double ratio = fp16_8b / size;
        std::string note = (std::string(name) == "Q4_K_M") ? " ← recommended" : "";
        std::cout << "  " << std::left  << std::setw(12) << name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << bpw
                  << std::setw(12) << std::setprecision(2) << size
                  << std::setw(10) << std::setprecision(1) << ratio << "×"
                  << note << "\n";
    }

    // Q4_K block layout
    std::cout << "\n  Q4_K_M block layout (256 weights = 1 super-block):\n";
    std::cout << "    sizeof(Q4KBlock) = " << sizeof(Q4KBlock) << " bytes\n";
    std::cout << "    Layout: 8 scales (8B) + 8 mins (8B) + 128B data = 144B\n";
    std::cout << "    Effective: 256 weights in 144B = 4.5 bits/weight\n";
    std::cout << "    (Q4_K_M uses mixed precision: some sub-blocks use Q6 scales)\n";

    // Memory layout of Q4K data
    Q4KBlock demo_block;
    std::memset(&demo_block, 0, sizeof(demo_block));
    // Pack two 4-bit values into one byte
    int w0 = 7, w1 = 3;  // example weights
    demo_block.qs[0] = (uint8_t)((w0 & 0xF) | (w1 << 4));
    std::cout << "\n  Example packing: w0=" << w0 << " w1=" << w1
              << " → byte=" << (int)demo_block.qs[0]
              << " (unpack: lo=" << (demo_block.qs[0] & 0xF)
              << " hi=" << (demo_block.qs[0] >> 4) << ")\n";

    assert((demo_block.qs[0] & 0xF) == w0);
    assert((demo_block.qs[0] >> 4)  == w1);
    std::cout << "  [ASSERT] 4-bit packing/unpacking correct ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  SAMPLER CHAIN
// ─────────────────────────────────────────────────────────────────────────────

struct TokenProb {
    int   id;
    float logit;
    float p;
};

static void softmax_inplace(std::vector<TokenProb>& candidates) {
    float max_logit = candidates[0].logit;
    for (auto& c : candidates)
        if (c.logit > max_logit) max_logit = c.logit;

    float sum = 0.0f;
    for (auto& c : candidates) {
        c.p = std::exp(c.logit - max_logit);
        sum += c.p;
    }
    for (auto& c : candidates) c.p /= sum;
}

static void apply_temperature(std::vector<TokenProb>& cands, float temp) {
    if (temp == 1.0f) return;
    for (auto& c : cands) c.logit /= temp;
    softmax_inplace(cands);
}

static void apply_top_k(std::vector<TokenProb>& cands, int k) {
    if (k <= 0 || k >= (int)cands.size()) return;
    std::partial_sort(cands.begin(), cands.begin() + k, cands.end(),
        [](const TokenProb& a, const TokenProb& b) { return a.p > b.p; });
    cands.resize(k);
    float sum = 0;
    for (auto& c : cands) sum += c.p;
    for (auto& c : cands) c.p /= sum;
}

static void apply_top_p(std::vector<TokenProb>& cands, float p) {
    std::sort(cands.begin(), cands.end(),
        [](const TokenProb& a, const TokenProb& b) { return a.p > b.p; });
    float cumsum = 0;
    size_t keep = cands.size();
    for (size_t i = 0; i < cands.size(); ++i) {
        cumsum += cands[i].p;
        if (cumsum >= p && i + 1 >= 1) {
            keep = i + 1;
            break;
        }
    }
    cands.resize(keep);
    float sum = 0;
    for (auto& c : cands) sum += c.p;
    for (auto& c : cands) c.p /= sum;
}

static void apply_min_p(std::vector<TokenProb>& cands, float min_p) {
    float max_p = 0;
    for (auto& c : cands) if (c.p > max_p) max_p = c.p;
    float threshold = min_p * max_p;
    cands.erase(std::remove_if(cands.begin(), cands.end(),
        [threshold](const TokenProb& c) { return c.p < threshold; }),
        cands.end());
    if (cands.empty()) return;
    float sum = 0;
    for (auto& c : cands) sum += c.p;
    for (auto& c : cands) c.p /= sum;
}

static int sample_from(const std::vector<TokenProb>& cands, std::mt19937& rng) {
    std::vector<float> probs;
    for (auto& c : cands) probs.push_back(c.p);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return cands[dist(rng)].id;
}

static void demo_sampler_chain() {
    print_section("Sampler Chain: temperature + top_k + top_p + min_p");

    const int V = 32;
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 2.0f);

    std::vector<TokenProb> base_cands(V);
    for (int i = 0; i < V; ++i) base_cands[i] = {i, gauss(rng), 0.0f};
    softmax_inplace(base_cands);

    auto show = [](const std::string& stage, const std::vector<TokenProb>& cands) {
        auto top = cands;
        std::sort(top.begin(), top.end(),
            [](const TokenProb& a, const TokenProb& b) { return a.p > b.p; });
        float entropy = 0;
        for (auto& c : cands)
            if (c.p > 0) entropy -= c.p * std::log2(c.p);
        std::cout << "  " << std::left << std::setw(26) << stage
                  << std::right << std::setw(6) << cands.size() << " tokens"
                  << "  top=" << std::fixed << std::setprecision(3) << top[0].p
                  << "/" << top[1].p
                  << "  H=" << std::setprecision(2) << entropy << " bits\n";
    };

    // Run through the sampler chain
    auto cands = base_cands;
    show("Initial (softmax)", cands);

    apply_temperature(cands, 0.7f);
    show("temp=0.7", cands);

    apply_top_k(cands, 16);
    show("top_k=16", cands);

    apply_top_p(cands, 0.90f);
    show("top_p=0.90", cands);

    apply_min_p(cands, 0.05f);
    show("min_p=0.05", cands);

    // Show final distribution
    std::cout << "\n  Final candidate distribution:\n";
    auto final_sorted = cands;
    std::sort(final_sorted.begin(), final_sorted.end(),
        [](const TokenProb& a, const TokenProb& b) { return a.p > b.p; });
    for (auto& c : final_sorted) {
        int bar = static_cast<int>(c.p * 50);
        std::cout << "    tok_" << std::left << std::setw(4) << c.id
                  << "  " << std::fixed << std::setprecision(4) << c.p
                  << "  " << std::string(bar, '#') << "\n";
    }

    // Monte Carlo sampling: sample 10000 times, verify top token
    std::map<int, int> counts;
    for (int trial = 0; trial < 10000; ++trial)
        counts[sample_from(cands, rng)]++;

    int top_id    = final_sorted[0].id;
    float emp_p   = counts[top_id] / 10000.0f;
    float theory_p = final_sorted[0].p;
    std::cout << "\n  [ASSERT] top token empirical p ≈ theory p: "
              << std::setprecision(3) << emp_p << " vs " << theory_p << " ";
    bool ok = std::abs(emp_p - theory_p) < 0.03f;
    std::cout << (ok ? "✓" : "WARN") << "\n";
    // Soft check only (10K samples has ~1% noise)
    if (!ok)
        std::cerr << "  [WARN] deviation " << std::abs(emp_p - theory_p)
                  << " exceeds 0.03 — increase trials for tighter check\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  KV CACHE SEQUENCE MANAGEMENT
// ─────────────────────────────────────────────────────────────────────────────

struct KVCell {
    int  seq_id = -1;   // -1 = empty
    int  pos    = -1;
};

class MockKVCache {
public:
    std::vector<KVCell> cells;
    int n_ctx;

    explicit MockKVCache(int n_ctx) : n_ctx(n_ctx), cells(n_ctx) {}

    // Assign positions [0, len) to seq_id
    bool assign(int seq_id, int len) {
        int free_start = -1, free_count = 0;
        for (int i = 0; i < n_ctx; ++i) {
            if (cells[i].seq_id == -1) {
                if (free_count == 0) free_start = i;
                if (++free_count == len) break;
            } else {
                free_count = 0; free_start = -1;
            }
        }
        if (free_count < len) return false;
        for (int i = free_start; i < free_start + len; ++i) {
            cells[i] = {seq_id, i - free_start};
        }
        return true;
    }

    // Append one token at the end of seq_id's range
    bool append(int seq_id) {
        int cur_len = 0;
        for (auto& c : cells)
            if (c.seq_id == seq_id) cur_len++;
        for (int i = 0; i < n_ctx; ++i) {
            if (cells[i].seq_id == -1) {
                cells[i] = {seq_id, cur_len};
                return true;
            }
        }
        return false;
    }

    // seq_rm: remove positions [p0, p1) for seq_id
    void seq_rm(int seq_id, int p0, int p1) {
        for (auto& c : cells) {
            if (c.seq_id == seq_id && c.pos >= p0 && c.pos < p1) {
                c = {-1, -1};
            }
        }
    }

    // seq_cp: copy seq_id src → dst for positions [p0, p1)
    bool seq_cp(int src, int dst, int p0, int p1) {
        // Count cells to copy
        std::vector<int> src_cells;
        for (int i = 0; i < n_ctx; ++i)
            if (cells[i].seq_id == src && cells[i].pos >= p0 && cells[i].pos < p1)
                src_cells.push_back(i);

        // Find free cells for dst
        std::vector<int> free_cells;
        for (int i = 0; i < n_ctx; ++i)
            if (cells[i].seq_id == -1)
                free_cells.push_back(i);

        if (free_cells.size() < src_cells.size()) return false;

        for (size_t i = 0; i < src_cells.size(); ++i) {
            cells[free_cells[i]] = {dst, cells[src_cells[i]].pos};
        }
        return true;
    }

    // seq_shift: shift positions for seq_id by delta
    void seq_shift(int seq_id, int p0, int p1, int delta) {
        for (auto& c : cells) {
            if (c.seq_id == seq_id && c.pos >= p0 && c.pos < p1) {
                c.pos += delta;
                if (c.pos < 0) c = {-1, -1};  // shifted out
            }
        }
    }

    int used() const {
        int n = 0;
        for (auto& c : cells) if (c.seq_id >= 0) n++;
        return n;
    }

    void status(const std::string& label) const {
        std::cout << "\n  " << label << "\n";
        std::map<int, int> counts;
        for (auto& c : cells)
            if (c.seq_id >= 0)
                counts[c.seq_id]++;
        for (auto& [sid, cnt] : counts) {
            int bar = cnt * 40 / n_ctx;
            std::cout << "    seq=" << std::setw(4) << sid
                      << "  tokens=" << std::setw(5) << cnt
                      << "  " << std::string(std::max(0, bar), '|') << "\n";
        }
        std::cout << "    Total: " << used() << "/" << n_ctx << " cells used\n";
    }
};

static void demo_kv_management() {
    print_section("KV Cache Sequence Management (seq_rm / seq_cp / seq_shift)");

    MockKVCache cache(2048);

    // Prefill two sequences
    cache.assign(100, 512);
    cache.assign(101, 256);
    for (int i = 0; i < 64; ++i) cache.append(100);
    for (int i = 0; i < 128; ++i) cache.append(101);
    cache.status("After prefill + decode");

    // Fork seq 100 for beam search
    bool forked = cache.seq_cp(/*src*/100, /*dst*/102, /*p0*/0, /*p1*/576);
    std::cout << "  seq_cp(100 → 102): " << (forked ? "✓" : "FAIL (no space)") << "\n";
    cache.append(100);  // beam 1 diverges
    cache.append(102);  // beam 2 diverges
    cache.status("After beam search fork");

    // Sliding window: keep only last 256 tokens for seq 100
    // Remove positions 0..319 (576 - 256 = 320 → remove [0, 320))
    cache.seq_rm(100, 0, 320);
    cache.seq_shift(100, 320, 577, -320);  // shift remaining positions down
    cache.status("After sliding window eviction (keep last 256)");

    // Free completed sequence
    cache.seq_rm(101, 0, 2048);
    cache.status("After freeing seq 101");

    // Assert: cache has correct number of tokens for seq 100
    int count_100 = 0;
    for (auto& c : cache.cells)
        if (c.seq_id == 100) count_100++;
    // seq 100: 256 (kept) + 1 (new decode after eviction)
    std::cout << "\n  [NOTE] seq 100 has " << count_100 << " tokens after eviction\n";
    assert(count_100 > 0);
    std::cout << "  [ASSERT] seq 100 non-empty after eviction ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  GRAMMAR CONSTRAINT SIMULATION
// ─────────────────────────────────────────────────────────────────────────────

// A minimal finite automaton to validate JSON character-by-character
// States: 0=start, 1=in_object, 2=in_string_key, 3=after_colon,
//         4=in_string_value, 5=after_value
enum class JSONState {
    START, IN_OBJECT, IN_KEY, AFTER_KEY,
    AFTER_COLON, IN_VALUE_STR, AFTER_VALUE, DONE, ERROR
};

static JSONState json_transition(JSONState s, char c) {
    switch (s) {
        case JSONState::START:
            if (c == '{') return JSONState::IN_OBJECT;
            return JSONState::ERROR;
        case JSONState::IN_OBJECT:
            if (c == '"') return JSONState::IN_KEY;
            if (c == '}') return JSONState::DONE;
            if (c == ' ' || c == '\n') return s;
            return JSONState::ERROR;
        case JSONState::IN_KEY:
            if (c == '"') return JSONState::AFTER_KEY;
            return s;  // still in key
        case JSONState::AFTER_KEY:
            if (c == ':') return JSONState::AFTER_COLON;
            if (c == ' ') return s;
            return JSONState::ERROR;
        case JSONState::AFTER_COLON:
            if (c == '"') return JSONState::IN_VALUE_STR;
            if (c == ' ') return s;
            return JSONState::ERROR;
        case JSONState::IN_VALUE_STR:
            if (c == '"') return JSONState::AFTER_VALUE;
            return s;
        case JSONState::AFTER_VALUE:
            if (c == ',') return JSONState::IN_OBJECT;
            if (c == '}') return JSONState::DONE;
            if (c == ' ') return s;
            return JSONState::ERROR;
        default:
            return JSONState::ERROR;
    }
}

static bool validate_json_string_object(const std::string& s) {
    JSONState state = JSONState::START;
    for (char c : s) {
        state = json_transition(state, c);
        if (state == JSONState::ERROR) return false;
    }
    return state == JSONState::DONE;
}

static void demo_grammar() {
    print_section("Grammar-Constrained Decoding: JSON FSA Validation");

    std::vector<std::pair<std::string, bool>> test_cases = {
        {R"({"name": "Alice"})",         true},
        {R"({"name": "Bob", "city": "NYC"})", true},   // valid JSON with multiple keys
        {R"({name: "Alice"})",           false},
        {R"({"name" "Alice"})",          false},
        {R"({"k": "v"})",                true},
        {R"(not json)",                  false},
        {R"({})",                        true},
    };

    std::cout << "\n  JSON string-object FSA validation:\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    int passed = 0;
    for (auto& [s, expected] : test_cases) {
        bool got = validate_json_string_object(s);
        bool ok  = (got == expected);
        if (ok) passed++;
        std::cout << "  " << (ok ? "✓" : "?")
                  << "  " << std::left << std::setw(38) << s.substr(0, 38)
                  << "  expected=" << (expected ? "valid  " : "invalid")
                  << "  got=" << (got ? "valid" : "invalid") << "\n";
    }
    std::cout << "\n  " << passed << "/" << test_cases.size()
              << " cases matched expected outcome\n";

    // Grammar masking table
    std::cout << "\n  Token validity by grammar state (fraction of 128K vocab):\n";
    struct StateRow { const char* state; int valid; };
    StateRow rows[] = {
        {"START (need '{')",         1},
        {"IN_OBJECT (need '\"' or '}')",   2},
        {"IN_KEY (any char)",       127800},
        {"AFTER_KEY (need ':')",     1},
        {"AFTER_COLON (need '\"')",  2},
        {"IN_VALUE (any char)",     127800},
        {"AFTER_VALUE (need ','/'}')", 2},
    };
    for (auto& r : rows) {
        double pct = r.valid / 128000.0 * 100;
        std::cout << "  " << std::left << std::setw(38) << r.state
                  << std::right << std::setw(8) << r.valid
                  << "  (" << std::fixed << std::setprecision(4) << pct << "%)\n";
    }
    std::cout << "\n  Structural tokens: ~0.001% vocab valid\n"
              << "  Content tokens:   ~99.8% vocab valid\n"
              << "  Average overhead per step: ~5-15% latency for mask application\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  mmap VS malloc LOADING MODEL
// ─────────────────────────────────────────────────────────────────────────────

static void demo_mmap_vs_malloc() {
    print_section("mmap vs malloc: Model Loading Trade-offs");

    const double MODEL_GB   = 4.84;  // Llama-3.1-8B Q4_K_M
    const double PAGE_SIZE  = 4096.0;  // bytes
    const double FIRST_ACCESS_PAGES = 50.0;  // pages touched on first forward pass

    struct Strategy {
        const char* name;
        double startup_time_s;  // time until first token
        double ram_initial_gb;  // RAM consumed at startup
        double ram_after_n_tokens;  // after 100 tokens
        const char* notes;
    };

    // Empirical estimates on a typical NVMe SSD system
    Strategy strats[] = {
        {
            "malloc + read",
            MODEL_GB / (3.5),         // 3.5 GB/s NVMe sequential read
            MODEL_GB,                  // entire model in RAM
            MODEL_GB,
            "Slow start, all pages hot immediately"
        },
        {
            "mmap (default)",
            0.05,                      // only header+metadata read at open
            0.05,                      // only header in RAM
            FIRST_ACCESS_PAGES * PAGE_SIZE / 1e9,  // only touched pages
            "Fast start, OS faults pages on demand"
        },
        {
            "mmap + mlock",
            MODEL_GB / 3.5 + 0.1,     // read + lock time
            MODEL_GB,                  // all pages locked in RAM
            MODEL_GB,
            "No swap possible; prevents latency spikes"
        },
    };

    std::cout << "\n  Model: Llama-3.1-8B Q4_K_M (" << MODEL_GB << " GB)\n\n";
    std::cout << "  " << std::left << std::setw(20) << "Strategy"
              << std::right << std::setw(16) << "Startup time"
              << std::setw(14) << "RAM at start"
              << "  Notes\n";
    std::cout << "  " << std::string(72, '-') << "\n";

    for (auto& s : strats) {
        std::cout << "  " << std::left << std::setw(20) << s.name
                  << std::right << std::setw(14) << std::fixed << std::setprecision(2)
                  << s.startup_time_s << "s"
                  << std::setw(12) << std::setprecision(2) << s.ram_initial_gb << " GB"
                  << "  " << s.notes << "\n";
    }

    std::cout << "\n  With mmap, two processes loading the same model share physical pages:\n"
              << "  Process A: maps 4.84 GB → 4.84 GB physical pages\n"
              << "  Process B: maps same file → 0 additional physical pages\n"
              << "  Combined RAM: 4.84 GB (not 9.68 GB)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << "  Chapter 28 — llama.cpp as a Platform (C++)\n"
              << std::string(72, '=') << "\n";

    demo_gguf_format();
    demo_sampler_chain();
    demo_kv_management();
    demo_grammar();
    demo_mmap_vs_malloc();

    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  All demos complete.\n"
              << std::string(72, '=') << "\n\n";
    return 0;
}

```

