# Chapter 10: Quantization Internals — Companion Code

## Python — `quantization_demo.py`

```python
"""
Chapter 10 — Quantization Internals: GGUF vs. vLLM
Companion code: quantization_demo.py

Sections:
  1. Precision formats: FP32 / FP16 / BF16 / INT8 / INT4 error analysis
  2. GGUF Q8_0 block encoder / decoder (bit-accurate)
  3. GGUF Q4_0 block encoder / decoder (nibble packing)
  4. GGUF Q4_K super-block encoder / decoder (two-level scales)
  5. GPTQ-style error propagation (simplified, no full Hessian)
  6. AWQ per-channel activation-aware scaling
  7. KV cache INT8 quantization and attention accuracy
  8. Roofline speedup predictions per quantization scheme

Run:
  python3 quantization_demo.py

No external dependencies beyond the standard library.
"""

from __future__ import annotations
import math
import random
import struct
import time
from dataclasses import dataclass
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def rmse(a: List[float], b: List[float]) -> float:
    n = len(a)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / n)

def max_abs_error(a: List[float], b: List[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))

def snr_db(original: List[float], reconstructed: List[float]) -> float:
    """Signal-to-noise ratio in dB. Higher = better."""
    signal_power = sum(x**2 for x in original) / len(original)
    noise_power  = sum((x-y)**2 for x, y in zip(original, reconstructed)) / len(original)
    if noise_power < 1e-30:
        return float('inf')
    return 10 * math.log10(signal_power / noise_power)

def random_weights(n: int, seed: int = 42,
                   dist: str = "normal") -> List[float]:
    """Generate synthetic weight values."""
    rng = random.Random(seed)
    if dist == "normal":
        # Box-Muller normal with std ~0.02 (typical weight scale)
        result = []
        while len(result) < n:
            u1 = rng.random() or 1e-10
            u2 = rng.random()
            z  = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            result.append(z * 0.02)
        return result[:n]
    elif dist == "uniform":
        return [rng.uniform(-1.0, 1.0) for _ in range(n)]
    else:
        raise ValueError(f"Unknown dist: {dist}")


# ──────────────────────────────────────────────────────────────────────────────
# §1  Precision Format Error Analysis
# ──────────────────────────────────────────────────────────────────────────────

def fp32_to_fp16_round_trip(x: float) -> float:
    """Simulate FP16 round-trip via struct pack/unpack."""
    packed = struct.pack('e', x)   # 'e' = half-precision float
    return struct.unpack('e', packed)[0]

def fp32_to_bf16_round_trip(x: float) -> float:
    """BF16 = FP32 with mantissa truncated to 7 bits."""
    packed = struct.pack('f', x)
    # Zero out the lower 16 bits of the mantissa (bytes 0 and 1 in little-endian)
    bf16_bytes = bytes([0, 0]) + packed[2:]
    return struct.unpack('f', bf16_bytes)[0]

def quantize_symmetric(x: float, scale: float, bits: int) -> int:
    """Symmetric integer quantization."""
    q_max = (1 << (bits - 1)) - 1   # e.g., 127 for INT8, 7 for INT4
    q = round(x / scale)
    return max(-q_max, min(q_max, q))

def dequantize_symmetric(q: int, scale: float) -> float:
    return q * scale

def section1_precision_formats():
    print("=" * 70)
    print("§1  Precision Format Error Analysis")
    print("=" * 70)

    n = 1024
    weights = random_weights(n, dist="normal")

    print(f"\n  {n} synthetic weights, distribution: N(0, 0.02²)\n")
    print(f"  {'Format':<12}  {'RMSE':>12}  {'Max error':>12}  "
          f"{'SNR (dB)':>10}  {'bpw':>5}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*5}")

    # BF16
    bf16 = [fp32_to_bf16_round_trip(w) for w in weights]
    print(f"  {'BF16':<12}  {rmse(weights,bf16):>12.6f}  "
          f"{max_abs_error(weights,bf16):>12.6f}  "
          f"{snr_db(weights,bf16):>10.2f}  {'16':>5}")

    # FP16
    fp16 = [fp32_to_fp16_round_trip(w) for w in weights]
    print(f"  {'FP16':<12}  {rmse(weights,fp16):>12.6f}  "
          f"{max_abs_error(weights,fp16):>12.6f}  "
          f"{snr_db(weights,fp16):>10.2f}  {'16':>5}")

    # INT8 symmetric
    max_abs = max(abs(w) for w in weights)
    scale8  = max_abs / 127.0
    int8    = [dequantize_symmetric(quantize_symmetric(w, scale8, 8), scale8)
               for w in weights]
    print(f"  {'INT8 symm':<12}  {rmse(weights,int8):>12.6f}  "
          f"{max_abs_error(weights,int8):>12.6f}  "
          f"{snr_db(weights,int8):>10.2f}  {'8':>5}")

    # INT4 symmetric
    scale4 = max_abs / 7.0
    int4   = [dequantize_symmetric(quantize_symmetric(w, scale4, 4), scale4)
              for w in weights]
    print(f"  {'INT4 symm':<12}  {rmse(weights,int4):>12.6f}  "
          f"{max_abs_error(weights,int4):>12.6f}  "
          f"{snr_db(weights,int4):>10.2f}  {'4':>5}")

    # INT4 block-wise (block_size=32) — should be better than global INT4
    block_size = 32
    int4_block = []
    for i in range(0, n, block_size):
        blk   = weights[i:i+block_size]
        bmax  = max(abs(w) for w in blk)
        bscale= bmax / 7.0 if bmax > 0 else 1.0
        int4_block.extend(
            dequantize_symmetric(quantize_symmetric(w, bscale, 4), bscale)
            for w in blk
        )
    print(f"  {'INT4 b=32':<12}  {rmse(weights,int4_block):>12.6f}  "
          f"{max_abs_error(weights,int4_block):>12.6f}  "
          f"{snr_db(weights,int4_block):>10.2f}  {'4.5':>5}")

    # INT2 block-wise (to show floor)
    int2_block = []
    for i in range(0, n, block_size):
        blk   = weights[i:i+block_size]
        bmax  = max(abs(w) for w in blk)
        bscale= bmax / 1.0 if bmax > 0 else 1.0   # 2-bit: [-1, 1]
        int2_block.extend(
            dequantize_symmetric(quantize_symmetric(w, bscale, 2), bscale)
            for w in blk
        )
    print(f"  {'INT2 b=32':<12}  {rmse(weights,int2_block):>12.6f}  "
          f"{max_abs_error(weights,int2_block):>12.6f}  "
          f"{snr_db(weights,int2_block):>10.2f}  {'2.5':>5}")

    # BF16 overflow demonstration
    print(f"\n  BF16 vs FP16 overflow test:")
    large_vals = [65000.0, 65504.0, 65505.0, 70000.0, 1e5]
    for v in large_vals:
        try:
            fp16_v = fp32_to_fp16_round_trip(v)
        except (OverflowError, struct.error):
            fp16_v = float('inf')
        bf16_v = fp32_to_bf16_round_trip(v)
        print(f"    x={v:>10.0f}:  FP16={fp16_v:>10}  BF16={bf16_v:>10.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# §2  GGUF Q8_0 Block Encoder / Decoder
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE_Q8_0 = 32

@dataclass
class Q8_0Block:
    """
    GGUF Q8_0 block layout (34 bytes):
      Bytes 0-1:  d = FP16 scale  (max_abs / 127)
      Bytes 2-33: 32 × INT8 quantized weights
    """
    scale: float          # FP16 scale (stored and restored via struct)
    quants: List[int]     # 32 × INT8 values in [-127, 127]

    def encode(self) -> bytes:
        """Pack into 34-byte wire format."""
        data  = struct.pack('e', self.scale)        # 2 bytes FP16
        data += struct.pack('32b', *self.quants)    # 32 bytes INT8
        return data

    @classmethod
    def decode(cls, data: bytes) -> "Q8_0Block":
        scale  = struct.unpack('e', data[0:2])[0]
        quants = list(struct.unpack('32b', data[2:34]))
        return cls(scale=scale, quants=quants)


def quantize_q8_0(weights: List[float]) -> List[Q8_0Block]:
    """Quantize a flat weight list into Q8_0 blocks."""
    n = len(weights)
    assert n % BLOCK_SIZE_Q8_0 == 0
    blocks = []
    for i in range(0, n, BLOCK_SIZE_Q8_0):
        blk   = weights[i:i + BLOCK_SIZE_Q8_0]
        max_a = max(abs(w) for w in blk)
        scale = max_a / 127.0 if max_a > 0 else 1.0
        # Simulate FP16 round-trip for scale storage
        scale_fp16 = struct.unpack('e', struct.pack('e', scale))[0]
        quants = [max(-127, min(127, round(w / scale_fp16))) for w in blk]
        blocks.append(Q8_0Block(scale=scale_fp16, quants=quants))
    return blocks


def dequantize_q8_0(blocks: List[Q8_0Block]) -> List[float]:
    result = []
    for b in blocks:
        result.extend(q * b.scale for q in b.quants)
    return result


def section2_q8_0():
    print("\n" + "=" * 70)
    print("§2  GGUF Q8_0: Block Encoder / Decoder")
    print("=" * 70)

    n = 256   # 8 blocks of 32
    weights = random_weights(n, dist="normal")

    blocks   = quantize_q8_0(weights)
    recon    = dequantize_q8_0(blocks)

    print(f"\n  n={n} weights, {len(blocks)} blocks of {BLOCK_SIZE_Q8_0}")
    print(f"  RMSE:         {rmse(weights, recon):.6f}")
    print(f"  Max error:    {max_abs_error(weights, recon):.6f}")
    print(f"  SNR:          {snr_db(weights, recon):.2f} dB")

    # Wire-format round-trip
    wire = b"".join(b.encode() for b in blocks)
    decoded_blocks = [Q8_0Block.decode(wire[i*34:(i+1)*34])
                      for i in range(len(blocks))]
    recon2 = dequantize_q8_0(decoded_blocks)

    print(f"\n  Wire-format round-trip ({len(wire)} bytes):")
    print(f"  Max round-trip error: {max_abs_error(recon, recon2):.2e}  "
          f"[should be ~0]")

    # Size breakdown
    fp16_bytes = n * 2
    q8_bytes   = len(wire)
    print(f"\n  Memory:")
    print(f"    FP16:  {fp16_bytes} bytes")
    print(f"    Q8_0:  {q8_bytes} bytes  "
          f"({100*q8_bytes/fp16_bytes:.1f}% of FP16, "
          f"{fp16_bytes/q8_bytes:.2f}× compression)")
    print(f"    bpw:   {q8_bytes*8/n:.2f}")

    # Show first block detail
    b0 = blocks[0]
    print(f"\n  Block 0 detail:")
    print(f"    scale (FP16): {b0.scale:.6f}")
    print(f"    quants:       {b0.quants[:8]} ...")
    print(f"    wire (hex):   {b0.encode().hex()}")


# ──────────────────────────────────────────────────────────────────────────────
# §3  GGUF Q4_0 Block Encoder / Decoder (nibble packing)
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE_Q4_0 = 32

@dataclass
class Q4_0Block:
    """
    GGUF Q4_0 block layout (18 bytes):
      Bytes 0-1:  d = FP16 scale
      Bytes 2-17: 16 bytes of packed nibbles (32 × 4-bit values)

    Packing: byte k = (q[2k] & 0xF) | ((q[2k+1] & 0xF) << 4)
    Values stored as unsigned [0..15]; signed value = q_u - 8
    """
    scale: float
    packed: bytes    # 16 bytes

    @classmethod
    def from_weights(cls, blk: List[float]) -> "Q4_0Block":
        max_a  = max(abs(w) for w in blk)
        scale  = max_a / 7.0 if max_a > 0 else 1.0
        scale  = struct.unpack('e', struct.pack('e', scale))[0]   # FP16
        nibbles = []
        for w in blk:
            q = max(-8, min(7, round(w / scale)))
            nibbles.append(q + 8)    # unsigned [0..15]
        # Pack two nibbles per byte
        packed = bytes(
            (nibbles[2*k] & 0xF) | ((nibbles[2*k+1] & 0xF) << 4)
            for k in range(16)
        )
        return cls(scale=scale, packed=packed)

    def dequantize(self) -> List[float]:
        result = []
        for byte in self.packed:
            lo = byte & 0xF
            hi = (byte >> 4) & 0xF
            result.append((lo - 8) * self.scale)
            result.append((hi - 8) * self.scale)
        return result

    def encode(self) -> bytes:
        return struct.pack('e', self.scale) + self.packed

    @classmethod
    def decode(cls, data: bytes) -> "Q4_0Block":
        scale  = struct.unpack('e', data[0:2])[0]
        packed = data[2:18]
        return cls(scale=scale, packed=packed)


def section3_q4_0():
    print("\n" + "=" * 70)
    print("§3  GGUF Q4_0: Nibble Packing Encoder / Decoder")
    print("=" * 70)

    n = 256
    weights = random_weights(n, dist="normal")

    blocks = [Q4_0Block.from_weights(weights[i:i+BLOCK_SIZE_Q4_0])
              for i in range(0, n, BLOCK_SIZE_Q4_0)]
    recon  = []
    for b in blocks:
        recon.extend(b.dequantize())

    print(f"\n  n={n} weights, {len(blocks)} blocks of {BLOCK_SIZE_Q4_0}")
    print(f"  RMSE:      {rmse(weights, recon):.6f}")
    print(f"  Max error: {max_abs_error(weights, recon):.6f}")
    print(f"  SNR:       {snr_db(weights, recon):.2f} dB")

    # Wire round-trip
    wire   = b"".join(b.encode() for b in blocks)
    blocks2= [Q4_0Block.decode(wire[i*18:(i+1)*18]) for i in range(len(blocks))]
    recon2 = []
    for b in blocks2:
        recon2.extend(b.dequantize())
    print(f"\n  Wire round-trip ({len(wire)} bytes): "
          f"max error = {max_abs_error(recon, recon2):.2e}")

    # Memory
    fp16_bytes = n * 2
    q4_bytes   = len(wire)
    print(f"\n  Memory:")
    print(f"    FP16:  {fp16_bytes} bytes")
    print(f"    Q4_0:  {q4_bytes} bytes  "
          f"({100*q4_bytes/fp16_bytes:.1f}% of FP16, "
          f"{fp16_bytes/q4_bytes:.2f}× compression)")
    print(f"    bpw:   {q4_bytes*8/n:.2f}")

    # Show nibble packing for first block
    b0 = blocks[0]
    print(f"\n  Block 0 packing detail:")
    print(f"    scale (FP16): {b0.scale:.6f}")
    all_nibbles = []
    for byte in b0.packed:
        all_nibbles.append(byte & 0xF)
        all_nibbles.append((byte >> 4) & 0xF)
    print(f"    nibbles [0..7]:  {all_nibbles[:8]}")
    print(f"    (signed) [0..7]: {[q-8 for q in all_nibbles[:8]]}")
    print(f"    packed bytes:    {b0.packed[:4].hex()} ...")


# ──────────────────────────────────────────────────────────────────────────────
# §4  GGUF Q4_K Super-Block Encoder / Decoder
# ──────────────────────────────────────────────────────────────────────────────

SUPER_BLOCK_SIZE = 256    # weights per super-block
INNER_BLOCK_SIZE =  32    # weights per inner block (8 per super-block)
N_INNER_BLOCKS   =   8    # SUPER_BLOCK_SIZE // INNER_BLOCK_SIZE

# 6-bit scale/min packed into 6 bytes (8 × 6-bit = 48 bits = 6 bytes)
SCALE_BITS = 6
SCALE_MAX  = (1 << SCALE_BITS) - 1   # 63

@dataclass
class Q4KSuperBlock:
    """
    GGUF Q4_K super-block layout (144 bytes per 256 weights):
      2 bytes:  d    = FP16 scale for scales  (super_scale)
      2 bytes:  dmin = FP16 scale for mins    (super_min, always positive)
      6 bytes:  scales[0..7]  (8 × 6-bit packed, unsigned)
      6 bytes:  mins[0..7]    (8 × 6-bit packed, unsigned magnitudes)
      128 bytes: data (256 × 4-bit packed, unsigned [0..15])
    Total: 144 bytes → 4.5 bpw

    Sign convention (mirroring llama.cpp ggml-quants.c):
      per-block scale:     block_scale = scales[b] * d       (positive)
      per-block min:       block_min   = -(mins[b] * dmin)   (negative, SUBTRACTED)
      quantize:   q = round((w - block_min) / block_scale)
                    = round((w + mins[b]*dmin) / block_scale)
      dequantize: w = q * block_scale + block_min
                    = q * block_scale - mins[b] * dmin

    The block min is always stored as its magnitude; reconstruction always subtracts it.
    This works because block min ≤ 0 for the convention used (shifted range [0, 15]).
    """
    super_scale: float   # d: scale for the 6-bit scale values
    super_min:   float   # dmin: scale for the 6-bit min magnitudes (positive)
    scales: List[int]    # 8 × 6-bit unsigned values
    mins:   List[int]    # 8 × 6-bit unsigned magnitudes of per-block mins
    data:   bytes        # 128 bytes of packed nibbles (unsigned [0..15])

    @classmethod
    def from_weights(cls, weights: List[float]) -> "Q4KSuperBlock":
        assert len(weights) == SUPER_BLOCK_SIZE

        # Step 1: compute per-block asymmetric scale and (positive) min magnitude
        block_scales = []
        block_min_mags = []    # |min(block)|  — always non-negative
        for b in range(N_INNER_BLOCKS):
            blk   = weights[b*INNER_BLOCK_SIZE : (b+1)*INNER_BLOCK_SIZE]
            lo    = min(blk)
            hi    = max(blk)
            span  = hi - lo
            scale = span / 15.0 if span > 1e-12 else 1.0
            # Store |min| as the magnitude; sign is implicit (always subtracted)
            block_scales.append(scale)
            block_min_mags.append(abs(lo))    # lo ≤ hi, lo is the min

        # Step 2: super-scale / super-min (positive FP16 values)
        max_scale = max(block_scales)
        max_min   = max(block_min_mags)
        super_scale = max_scale / SCALE_MAX if max_scale > 1e-12 else 1.0
        super_min   = max_min   / SCALE_MAX if max_min   > 1e-12 else 1.0

        # FP16 round-trip
        super_scale = struct.unpack('e', struct.pack('e', super_scale))[0]
        super_min   = struct.unpack('e', struct.pack('e', super_min))[0]

        scales_q = [max(0, min(SCALE_MAX, round(s / super_scale)))
                    for s in block_scales]
        mins_q   = [max(0, min(SCALE_MAX, round(m / super_min)))
                    for m in block_min_mags]

        # Step 3: quantize weights using recovered scale and min
        #   w ≈ q × block_scale − mins[b] × super_min
        #   q = round((w + mins[b] × super_min) / block_scale)
        nibbles = []
        for b in range(N_INNER_BLOCKS):
            rec_scale = scales_q[b] * super_scale
            rec_min   = mins_q[b]   * super_min    # positive magnitude
            blk       = weights[b*INNER_BLOCK_SIZE : (b+1)*INNER_BLOCK_SIZE]
            for w in blk:
                if rec_scale > 1e-12:
                    q = round((w + rec_min) / rec_scale)
                else:
                    q = 0
                nibbles.append(max(0, min(15, q)))

        # Pack nibbles → 128 bytes
        data = bytes(
            (nibbles[2*k] & 0xF) | ((nibbles[2*k+1] & 0xF) << 4)
            for k in range(128)
        )

        return cls(
            super_scale=super_scale,
            super_min=super_min,
            scales=scales_q,
            mins=mins_q,
            data=data,
        )

    def dequantize(self) -> List[float]:
        """
        Dequantize: w = q × block_scale − mins[b] × super_min
        """
        result = []
        for b in range(N_INNER_BLOCKS):
            rec_scale = self.scales[b] * self.super_scale
            rec_min   = self.mins[b]   * self.super_min   # positive magnitude
            start_byte = b * 16
            for k in range(16):
                byte = self.data[start_byte + k]
                lo = byte & 0xF
                hi = (byte >> 4) & 0xF
                result.append(lo * rec_scale - rec_min)
                result.append(hi * rec_scale - rec_min)
        return result

    def encode(self) -> bytes:
        """Pack the super-block into 144 bytes."""
        out  = struct.pack('e', self.super_scale)
        out += struct.pack('e', self.super_min)
        out += _pack_6bit(self.scales)
        out += _pack_6bit(self.mins)
        out += self.data
        return out  # 144 bytes

    @classmethod
    def decode(cls, data: bytes) -> "Q4KSuperBlock":
        super_scale = struct.unpack('e', data[0:2])[0]
        super_min   = struct.unpack('e', data[2:4])[0]
        scales      = _unpack_6bit(data[4:10])
        mins        = _unpack_6bit(data[10:16])
        payload     = data[16:144]
        return cls(super_scale=super_scale, super_min=super_min,
                   scales=scales, mins=mins, data=payload)


def _pack_6bit(values: List[int]) -> bytes:
    """Pack 8 × 6-bit values into 6 bytes (48 bits)."""
    assert len(values) == 8
    bits = 0
    for v in values:
        bits = (bits << 6) | (v & 0x3F)
    return bits.to_bytes(6, 'big')


def _unpack_6bit(data: bytes) -> List[int]:
    """Unpack 6 bytes into 8 × 6-bit values."""
    bits = int.from_bytes(data, 'big')
    values = []
    for _ in range(8):
        values.append(bits & 0x3F)
        bits >>= 6
    return list(reversed(values))


def section4_q4k():
    print("\n" + "=" * 70)
    print("§4  GGUF Q4_K: Super-Block Encoder / Decoder")
    print("=" * 70)

    n = 512   # 2 super-blocks
    weights = random_weights(n, dist="normal")

    blocks = [Q4KSuperBlock.from_weights(weights[i:i+SUPER_BLOCK_SIZE])
              for i in range(0, n, SUPER_BLOCK_SIZE)]

    recon = []
    for b in blocks:
        recon.extend(b.dequantize())

    print(f"\n  n={n} weights, {len(blocks)} super-blocks of {SUPER_BLOCK_SIZE}")
    print(f"  RMSE:      {rmse(weights, recon):.6f}")
    print(f"  Max error: {max_abs_error(weights, recon):.6f}")
    print(f"  SNR:       {snr_db(weights, recon):.2f} dB")

    # Wire round-trip
    wire = b"".join(b.encode() for b in blocks)
    assert len(wire) == len(blocks) * 144, f"Expected {len(blocks)*144} bytes, got {len(wire)}"
    blocks2 = [Q4KSuperBlock.decode(wire[i*144:(i+1)*144])
               for i in range(len(blocks))]
    recon2  = []
    for b in blocks2:
        recon2.extend(b.dequantize())
    print(f"\n  Wire round-trip ({len(wire)} bytes): "
          f"max error = {max_abs_error(recon, recon2):.2e}")

    # Memory
    fp16_bytes = n * 2
    q4k_bytes  = len(wire)
    print(f"\n  Memory:")
    print(f"    FP16:   {fp16_bytes} bytes")
    print(f"    Q4_K:   {q4k_bytes} bytes  "
          f"({100*q4k_bytes/fp16_bytes:.1f}% of FP16, "
          f"{fp16_bytes/q4k_bytes:.2f}× compression)")
    print(f"    bpw:    {q4k_bytes*8/n:.2f}")

    # Super-block structure detail
    b0 = blocks[0]
    print(f"\n  Super-block 0 detail:")
    print(f"    super_scale: {b0.super_scale:.6f}")
    print(f"    super_min:   {b0.super_min:.6f}")
    print(f"    scales (6-bit): {b0.scales}")
    print(f"    mins   (6-bit): {b0.mins}")
    print(f"    recovered per-block scales: "
          f"{[round(s*b0.super_scale,5) for s in b0.scales]}")

    # Q4_0 vs Q4_K accuracy comparison
    q4_0_blocks = [Q4_0Block.from_weights(weights[i:i+BLOCK_SIZE_Q4_0])
                   for i in range(0, n, BLOCK_SIZE_Q4_0)]
    recon_q4_0  = []
    for b in q4_0_blocks:
        recon_q4_0.extend(b.dequantize())

    print(f"\n  Q4_0 vs Q4_K accuracy (same {n} weights):")
    print(f"    Q4_0 RMSE:  {rmse(weights, recon_q4_0):.6f}  "
          f"SNR: {snr_db(weights, recon_q4_0):.2f} dB")
    print(f"    Q4_K RMSE:  {rmse(weights, recon):.6f}  "
          f"SNR: {snr_db(weights, recon):.2f} dB")
    print(f"    Q4_K advantage: "
          f"{snr_db(weights,recon)-snr_db(weights,recon_q4_0):+.2f} dB SNR")


# ──────────────────────────────────────────────────────────────────────────────
# §5  GPTQ-Style Error Propagation (simplified)
# ──────────────────────────────────────────────────────────────────────────────

def gptq_quantize_row(row: List[float],
                       scale: float,
                       bits: int = 4) -> Tuple[List[int], List[float]]:
    """
    Simplified GPTQ: quantize a weight row column by column with
    error propagation.

    Real GPTQ uses H^{-1} from the Hessian to weight the error correction.
    Here we propagate error uniformly to remaining columns (no Hessian),
    which demonstrates the principle without requiring calibration data.

    Returns:
      (quantized_ints, dequantized_floats)
    """
    n = len(row)
    q_max  = (1 << (bits - 1)) - 1
    result = list(row)    # copy — we'll modify in-place
    quants = []

    for j in range(n):
        # Quantize column j
        q = max(-q_max, min(q_max, round(result[j] / scale)))
        quants.append(q)
        # Error at column j
        err = result[j] - q * scale
        # Propagate error uniformly to remaining columns
        # (real GPTQ: weight by H_inv row)
        if j + 1 < n:
            correction = err / (n - j - 1)
            for k in range(j + 1, n):
                result[k] += correction

    dequant = [q * scale for q in quants]
    return quants, dequant


def rtn_quantize_row(row: List[float],
                      scale: float,
                      bits: int = 4) -> List[float]:
    """Round-to-nearest (no error propagation) — baseline for GPTQ comparison."""
    q_max = (1 << (bits - 1)) - 1
    return [max(-q_max, min(q_max, round(w / scale))) * scale for w in row]


def section5_gptq():
    print("\n" + "=" * 70)
    print("§5  GPTQ-Style Error Propagation (simplified)")
    print("=" * 70)

    n = 256
    weights = random_weights(n, dist="normal")
    max_abs = max(abs(w) for w in weights)
    scale   = max_abs / 7.0   # INT4 symmetric

    # RTN baseline
    rtn = rtn_quantize_row(weights, scale, bits=4)

    # GPTQ-style
    _, gptq = gptq_quantize_row(weights, scale, bits=4)

    print(f"\n  n={n} weights, INT4, scale={scale:.6f}")
    print(f"\n  {'Method':<20}  {'RMSE':>12}  {'SNR (dB)':>10}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*10}")
    print(f"  {'RTN (no propagat.)':<20}  "
          f"{rmse(weights,rtn):>12.6f}  {snr_db(weights,rtn):>10.2f}")
    print(f"  {'GPTQ (err. prop.)':<20}  "
          f"{rmse(weights,gptq):>12.6f}  {snr_db(weights,gptq):>10.2f}")
    print(f"\n  GPTQ advantage: "
          f"{snr_db(weights,gptq)-snr_db(weights,rtn):+.2f} dB SNR")
    print(f"\n  Note: real GPTQ uses Hessian weighting; this demo uses uniform")
    print(f"  propagation. The real gain is typically 2–4 dB higher.")


# ──────────────────────────────────────────────────────────────────────────────
# §6  AWQ Per-Channel Activation-Aware Scaling
# ──────────────────────────────────────────────────────────────────────────────

def awq_find_scale(act_scales: List[float], alpha: float = 0.5) -> List[float]:
    """
    AWQ per-channel scale: s[c] = act_scale[c]^alpha
    alpha typically tuned to 0.5 for most models.
    """
    return [a ** alpha for a in act_scales]


def awq_quantize(W: List[List[float]],
                  act_scales: List[float],
                  bits: int = 4,
                  alpha: float = 0.5) -> Tuple[List[List[float]], List[float]]:
    """
    AWQ quantization:
      1. Compute per-channel scale s = act_scales^alpha
      2. Scale weights: W'[:, c] = W[:, c] × s[c]
      3. Quantize W' with RTN
      4. At runtime: activation is divided by s[c] (fused into norm)

    Returns:
      dequant_W: reconstructed weight matrix
      s:         per-channel scales (for runtime compensation)
    """
    n_rows = len(W)
    n_cols = len(W[0])
    s = awq_find_scale(act_scales, alpha)

    dequant_W = []
    for row in W:
        # Scale weight row by s
        scaled = [w * s[c] for c, w in enumerate(row)]
        # Find per-row scale for quantization
        max_a  = max(abs(v) for v in scaled)
        qscale = max_a / ((1 << (bits - 1)) - 1) if max_a > 0 else 1.0
        # RTN quantize the scaled row
        q_max = (1 << (bits - 1)) - 1
        quants = [max(-q_max, min(q_max, round(v / qscale))) for v in scaled]
        # Dequantize (still in scaled space)
        dq_scaled = [q * qscale for q in quants]
        # Undo the scaling (recover original weight space)
        dequant_W.append([v / s[c] for c, v in enumerate(dq_scaled)])

    return dequant_W, s


def section6_awq():
    print("\n" + "=" * 70)
    print("§6  AWQ: Activation-Aware Per-Channel Scaling")
    print("=" * 70)

    rng = random.Random(42)
    n_rows, n_cols = 16, 64

    # Weight matrix
    W = [[rng.gauss(0, 0.02) for _ in range(n_cols)] for _ in range(n_rows)]

    # Activation scales: simulate large activations on a few "salient" channels
    act_scales = [rng.uniform(0.01, 0.1) for _ in range(n_cols)]
    # Make 5% of channels "salient" (high activation magnitude)
    salient = rng.sample(range(n_cols), n_cols // 20)
    for c in salient:
        act_scales[c] *= 20.0

    # RTN baseline (no AWQ)
    def rtn_matrix(W, bits=4):
        result = []
        for row in W:
            max_a  = max(abs(w) for w in row)
            scale  = max_a / ((1 << (bits-1)) - 1) if max_a > 0 else 1.0
            q_max  = (1 << (bits-1)) - 1
            result.append([max(-q_max, min(q_max, round(w/scale))) * scale
                           for w in row])
        return result

    rtn_W = rtn_matrix(W)
    awq_W, s = awq_quantize(W, act_scales, bits=4)

    # Compute errors
    def mat_rmse(A, B):
        flat_a = [v for row in A for v in row]
        flat_b = [v for row in B for v in row]
        return rmse(flat_a, flat_b)

    def weighted_mat_rmse(A, B, act_scales):
        """RMSE weighted by activation scale — what AWQ optimizes."""
        err = 0.0
        n   = 0
        for row_a, row_b in zip(A, B):
            for c, (a, b) in enumerate(zip(row_a, row_b)):
                err += (act_scales[c] * (a - b)) ** 2
                n   += 1
        return math.sqrt(err / n)

    print(f"\n  Weight matrix: {n_rows}×{n_cols}, INT4")
    print(f"  Salient channels ({len(salient)}/{n_cols}): "
          f"{sorted(salient)[:5]}...")
    print(f"  Max activation scale: {max(act_scales):.3f}")
    print(f"  Mean activation scale: {sum(act_scales)/len(act_scales):.4f}")

    print(f"\n  {'Method':<15}  {'RMSE':>10}  {'Weighted RMSE':>15}")
    print(f"  {'─'*15}  {'─'*10}  {'─'*15}")
    print(f"  {'RTN':<15}  {mat_rmse(W, rtn_W):>10.6f}  "
          f"{weighted_mat_rmse(W, rtn_W, act_scales):>15.6f}")
    print(f"  {'AWQ':<15}  {mat_rmse(W, awq_W):>10.6f}  "
          f"{weighted_mat_rmse(W, awq_W, act_scales):>15.6f}")

    print(f"\n  AWQ improves weighted RMSE by "
          f"{100*(1 - weighted_mat_rmse(W,awq_W,act_scales) / weighted_mat_rmse(W,rtn_W,act_scales)):.1f}%")
    print(f"  (RTN is better on raw RMSE; AWQ optimizes the activation-weighted error)")

    # Per-channel scale example
    print(f"\n  AWQ per-channel scales s[c] = act_scale[c]^0.5 (first 8 channels):")
    for c in range(8):
        marker = " ← salient" if c in salient else ""
        print(f"    c={c}: act_scale={act_scales[c]:.4f}  "
              f"awq_scale={s[c]:.4f}{marker}")


# ──────────────────────────────────────────────────────────────────────────────
# §7  KV Cache INT8 Quantization and Attention Accuracy
# ──────────────────────────────────────────────────────────────────────────────

def softmax(x: List[float]) -> List[float]:
    max_x = max(x)
    exp_x = [math.exp(v - max_x) for v in x]
    s = sum(exp_x)
    return [v / s for v in exp_x]


def attention_scores(Q: List[float], K_all: List[List[float]],
                     d_head: int) -> List[float]:
    """Compute attention scores for one query against all keys."""
    scale = 1.0 / math.sqrt(d_head)
    scores = []
    for k in K_all:
        dot = sum(q * ki for q, ki in zip(Q, k))
        scores.append(dot * scale)
    return softmax(scores)


def quantize_kv_int8(vectors: List[List[float]]) -> Tuple[List[List[int]], List[float]]:
    """
    Per-token INT8 quantization of K or V vectors.
    Returns (quantized_vectors, per_token_scales).
    """
    quant_vecs = []
    scales     = []
    for vec in vectors:
        max_a = max(abs(v) for v in vec) or 1.0
        scale = max_a / 127.0
        q_vec = [max(-127, min(127, round(v / scale))) for v in vec]
        quant_vecs.append(q_vec)
        scales.append(scale)
    return quant_vecs, scales


def dequantize_kv(quant_vecs: List[List[int]],
                   scales: List[float]) -> List[List[float]]:
    return [[q * s for q in vec]
            for vec, s in zip(quant_vecs, scales)]


def section7_kv_quant():
    print("\n" + "=" * 70)
    print("§7  KV Cache INT8 Quantization and Attention Accuracy")
    print("=" * 70)

    rng   = random.Random(42)
    d_head = 128
    seq_len = 64

    # Generate K and V tensors (float)
    K_fp = [[rng.gauss(0, 0.1) for _ in range(d_head)] for _ in range(seq_len)]
    V_fp = [[rng.gauss(0, 0.1) for _ in range(d_head)] for _ in range(seq_len)]

    # Query (current token)
    Q = [rng.gauss(0, 0.1) for _ in range(d_head)]

    # Reference attention with FP32 K, V
    scores_ref = attention_scores(Q, K_fp, d_head)
    # Weighted sum of V
    output_ref = [sum(scores_ref[t] * V_fp[t][i]
                      for t in range(seq_len))
                  for i in range(d_head)]

    # INT8 quantized K, V
    K_q8, K_scales = quantize_kv_int8(K_fp)
    V_q8, V_scales = quantize_kv_int8(V_fp)
    K_dq = dequantize_kv(K_q8, K_scales)
    V_dq = dequantize_kv(V_q8, V_scales)

    scores_int8 = attention_scores(Q, K_dq, d_head)
    output_int8 = [sum(scores_int8[t] * V_dq[t][i]
                       for t in range(seq_len))
                   for i in range(d_head)]

    # INT4 quantized K, V
    def quantize_kv_int4(vecs):
        quant_vecs, scales = [], []
        for vec in vecs:
            max_a = max(abs(v) for v in vec) or 1.0
            scale = max_a / 7.0
            q_vec = [max(-8, min(7, round(v / scale))) for v in vec]
            quant_vecs.append(q_vec)
            scales.append(scale)
        return quant_vecs, scales

    K_q4, K4_scales = quantize_kv_int4(K_fp)
    V_q4, V4_scales = quantize_kv_int4(V_fp)
    K_dq4 = dequantize_kv(K_q4, K4_scales)
    V_dq4 = dequantize_kv(V_q4, V4_scales)

    scores_int4 = attention_scores(Q, K_dq4, d_head)
    output_int4 = [sum(scores_int4[t] * V_dq4[t][i]
                       for t in range(seq_len))
                   for i in range(d_head)]

    print(f"\n  seq_len={seq_len}, d_head={d_head}")
    print(f"\n  Attention output error vs FP32 reference:\n")
    print(f"  {'Format':<12}  {'Output RMSE':>12}  {'Score KL-div':>14}  "
          f"{'bpw':>5}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*14}  {'─'*5}")

    def kl_div(p, q):
        return sum(pi * math.log(pi / (qi + 1e-12) + 1e-12)
                   for pi, qi in zip(p, q) if pi > 0)

    print(f"  {'FP32 (ref)':<12}  {'0.000000':>12}  {'0.000000':>14}  {'32':>5}")
    print(f"  {'INT8 KV':<12}  "
          f"{rmse(output_ref, output_int8):>12.6f}  "
          f"{kl_div(scores_ref, scores_int8):>14.8f}  {'8':>5}")
    print(f"  {'INT4 KV':<12}  "
          f"{rmse(output_ref, output_int4):>12.6f}  "
          f"{kl_div(scores_ref, scores_int4):>14.8f}  {'4':>5}")

    # Memory savings
    fp16_kv = seq_len * d_head * 2 * 2   # K+V, BF16
    int8_kv = seq_len * d_head * 1 * 2   # K+V, INT8
    int4_kv = seq_len * d_head // 2 * 2  # K+V, INT4 packed
    print(f"\n  KV cache memory (seq_len={seq_len}, d_head={d_head}):")
    print(f"    BF16: {fp16_kv} bytes  (1.0×)")
    print(f"    INT8: {int8_kv} bytes  ({fp16_kv/int8_kv:.1f}× reduction)")
    print(f"    INT4: {int4_kv} bytes  ({fp16_kv/int4_kv:.1f}× reduction)")
    print(f"\n  [COMMON TRAP] INT4 KV shows higher error than INT8.")
    print(f"  For long contexts this error accumulates — use INT8 as the floor.")


# ──────────────────────────────────────────────────────────────────────────────
# §8  Roofline Speedup Predictions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareSpec:
    name: str
    peak_tflops: float    # BF16/FP16
    hbm_bw_gbs: float     # GB/s

A100_80GB = HardwareSpec("A100 80GB", 312.0, 2000.0)
H100_SXM  = HardwareSpec("H100 SXM5", 1979.0, 3350.0)
RTX4090   = HardwareSpec("RTX 4090",  330.0, 1008.0)

@dataclass
class QuantScheme:
    name: str
    bpw: float                # bits per weight
    dequant_flops_per_w: float# extra FLOPs for dequant (approximate)
    practical_fraction: float # fraction of theoretical speedup achieved

SCHEMES = [
    QuantScheme("BF16 (baseline)",  16.0, 0.0,  1.00),
    QuantScheme("FP8 (H100)",        8.0, 0.1,  0.85),
    QuantScheme("Q8_0",              8.5, 0.5,  0.80),
    QuantScheme("Q6_K",              6.6, 1.0,  0.75),
    QuantScheme("Q5_K_M",            5.7, 1.5,  0.72),
    QuantScheme("Q4_K_M",            4.5, 2.1,  0.70),
    QuantScheme("Q3_K_M",            3.4, 3.0,  0.65),
    QuantScheme("Q2_K",              2.6, 4.0,  0.60),
]

def roofline_speedup(scheme: QuantScheme, hw: HardwareSpec,
                      n_weights: float, batch_size: int = 1) -> dict:
    """
    Estimate decode step speedup from quantization.
    Model: time ≈ (bytes_read + dequant_time) / bandwidth
    """
    bf16_bytes  = n_weights * 2.0
    quant_bytes = n_weights * scheme.bpw / 8.0

    bf16_time_ms = bf16_bytes / (hw.hbm_bw_gbs * 1e9) * 1000

    # Quantized: read fewer bytes + pay dequant FLOPs
    dequant_flops = n_weights * scheme.dequant_flops_per_w * batch_size
    dequant_time_ms = dequant_flops / (hw.peak_tflops * 1e12) * 1000
    bw_time_ms = quant_bytes / (hw.hbm_bw_gbs * 1e9) * 1000
    quant_time_ms = bw_time_ms + dequant_time_ms

    theoretical_speedup = bf16_time_ms / quant_time_ms
    practical_speedup   = theoretical_speedup * scheme.practical_fraction

    return {
        "bf16_time_ms":      bf16_time_ms,
        "bw_time_ms":        bw_time_ms,
        "dequant_time_ms":   dequant_time_ms,
        "quant_time_ms":     quant_time_ms,
        "theoretical_speedup": theoretical_speedup,
        "practical_speedup":   practical_speedup,
    }


def section8_roofline():
    print("\n" + "=" * 70)
    print("§8  Roofline Speedup Predictions per Quantization Scheme")
    print("=" * 70)

    # LLaMA 3 8B: ~7.03B non-embedding weights
    n_weights_8b = 7.03e9

    for hw in [A100_80GB, H100_SXM, RTX4090]:
        print(f"\n  Hardware: {hw.name}  "
              f"(peak {hw.peak_tflops:.0f} TFLOP/s, "
              f"HBM {hw.hbm_bw_gbs:.0f} GB/s)")
        print(f"  Model: LLaMA 3 8B, B=1 (decode)\n")
        print(f"  {'Scheme':<18}  {'bpw':>5}  {'BW time':>9}  "
              f"{'Dequant':>9}  {'Theory':>8}  {'Practical':>10}")
        print(f"  {'─'*18}  {'─'*5}  {'─'*9}  "
              f"{'─'*9}  {'─'*8}  {'─'*10}")

        for s in SCHEMES:
            r = roofline_speedup(s, hw, n_weights_8b, batch_size=1)
            print(f"  {s.name:<18}  {s.bpw:>5.1f}  "
                  f"{r['bw_time_ms']:>8.2f}ms  "
                  f"{r['dequant_time_ms']*1000:>7.2f}µs  "
                  f"{r['theoretical_speedup']:>7.2f}×  "
                  f"{r['practical_speedup']:>9.2f}×")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    section1_precision_formats()
    section2_q8_0()
    section3_q4_0()
    section4_q4k()
    section5_gptq()
    section6_awq()
    section7_kv_quant()
    section8_roofline()
    print("\n" + "=" * 70)
    print("All sections complete.")
    print("=" * 70)

```

## C++ — `quantization_demo.cpp`

```cpp
/*
 * Chapter 10 — Quantization Internals: GGUF vs. vLLM
 * Companion code: quantization_demo.cpp
 *
 * Sections:
 *   1. Precision format error analysis (FP16, BF16, INT8, INT4)
 *   2. GGUF Q8_0 block encoder / decoder (bit-accurate)
 *   3. GGUF Q4_0 block encoder / decoder (nibble packing)
 *   4. GGUF Q4_K super-block encoder / decoder (two-level scales)
 *   5. GPTQ-style error propagation (simplified)
 *   6. AWQ per-channel activation-aware scaling
 *   7. KV cache INT8 quantization and attention accuracy
 *   8. Roofline speedup predictions per quantization scheme
 *
 * Compile:
 *   g++ -O2 -std=c++17 -o quantization_demo quantization_demo.cpp && ./quantization_demo
 *
 * No external dependencies.
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::string fmt(double v, int prec = 2) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}
static std::string pad_right(const std::string& s, int w) {
    if ((int)s.size() >= w) return s.substr(0, w);
    return s + std::string(w - s.size(), ' ');
}
static std::string pad_left(const std::string& s, int w) {
    if ((int)s.size() >= w) return s.substr(0, w);
    return std::string(w - s.size(), ' ') + s;
}

// Minimal LCG RNG — deterministic, no <random> needed
struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed = 42) : state(seed) {}
    float next_uniform() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float)(state >> 33) / (float)(1u << 31);
    }
    // Box-Muller normal with given std
    float next_normal(float mu = 0.f, float sigma = 0.02f) {
        float u1 = next_uniform() + 1e-10f;
        float u2 = next_uniform();
        float z  = std::sqrt(-2.f * std::log(u1))
                   * std::cos(2.f * (float)M_PI * u2);
        return mu + sigma * z;
    }
};

static std::vector<float> make_weights(int n, float sigma = 0.02f,
                                        uint64_t seed = 42) {
    Lcg rng(seed);
    std::vector<float> w(n);
    for (auto& v : w) v = rng.next_normal(0.f, sigma);
    return w;
}

static double rmse(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s / a.size());
}

static double max_abs_error(const std::vector<float>& a,
                             const std::vector<float>& b) {
    double m = 0;
    for (int i = 0; i < (int)a.size(); i++)
        m = std::max(m, std::abs((double)(a[i] - b[i])));
    return m;
}

static double snr_db(const std::vector<float>& orig,
                      const std::vector<float>& recon) {
    double sp = 0, np = 0;
    for (int i = 0; i < (int)orig.size(); i++) {
        sp += (double)orig[i] * orig[i];
        double d = orig[i] - recon[i];
        np += d * d;
    }
    sp /= orig.size(); np /= orig.size();
    if (np < 1e-30) return 999.0;
    return 10.0 * std::log10(sp / np);
}

// ─────────────────────────────────────────────────────────────────────────────
// FP16 / BF16 helpers (no hardware dependency)
// ─────────────────────────────────────────────────────────────────────────────

// FP16 round-trip: encode to 16-bit half, decode back
static float fp32_to_fp16_rt(float x) {
    // Use standard bit manipulation for IEEE 754 half-precision
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    uint32_t sign     = (bits >> 31) & 1;
    int32_t  exp32    = (int32_t)((bits >> 23) & 0xFF) - 127;
    uint32_t mant32   = bits & 0x7FFFFF;

    if (exp32 > 15) {
        // Overflow → infinity
        uint16_t h = (uint16_t)((sign << 15) | 0x7C00);
        uint32_t b2 = (uint32_t)(sign << 31)
                    | (0xFF << 23);  // inf in FP32
        float r; std::memcpy(&r, &b2, 4); return r;
    }
    if (exp32 < -14) {
        // Underflow / denormal
        int shift = -14 - exp32;
        if (shift > 10) return 0.f;
        uint16_t mant16 = (uint16_t)((mant32 | 0x800000) >> (13 + shift));
        uint32_t b2 = (sign << 31)
                    | ((uint32_t)mant16 << 13);
        float r; std::memcpy(&r, &b2, 4); return r;
    }
    uint16_t exp16  = (uint16_t)(exp32 + 15);
    uint16_t mant16 = (uint16_t)(mant32 >> 13);
    // Encode then decode
    uint16_t h = (uint16_t)((sign << 15) | (exp16 << 10) | mant16);
    // Decode back to FP32
    uint32_t s2   = (h >> 15) & 1;
    uint32_t e2   = (h >> 10) & 0x1F;
    uint32_t m2   = h & 0x3FF;
    uint32_t b2;
    if (e2 == 0) {
        b2 = (s2 << 31) | (m2 << 13);
    } else if (e2 == 31) {
        b2 = (s2 << 31) | (0xFF << 23) | (m2 << 13);
    } else {
        b2 = (s2 << 31) | ((e2 + 112) << 23) | (m2 << 13);
    }
    float r; std::memcpy(&r, &b2, 4); return r;
}

// BF16 round-trip: zero lower 16 mantissa bits
static float fp32_to_bf16_rt(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    bits &= 0xFFFF0000u;    // keep top 16 bits (sign + 8-bit exp + 7 mant bits)
    float r; std::memcpy(&r, &bits, 4); return r;
}

// Encode a float as FP16 bits (for scale storage)
static uint16_t float_to_fp16_bits(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    uint32_t sign  = (bits >> 31) & 1;
    int32_t  exp32 = (int32_t)((bits >> 23) & 0xFF) - 127;
    uint32_t mant  = bits & 0x7FFFFF;
    if (exp32 > 15)  return (uint16_t)((sign << 15) | 0x7C00);
    if (exp32 < -14) return (uint16_t)(sign << 15);
    uint16_t e16 = (uint16_t)(exp32 + 15);
    uint16_t m16 = (uint16_t)(mant >> 13);
    return (uint16_t)((sign << 15) | (e16 << 10) | m16);
}

static float fp16_bits_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t b;
    if (e == 0)       b = (s << 31) | (m << 13);
    else if (e == 31) b = (s << 31) | (0xFF << 23) | (m << 13);
    else              b = (s << 31) | ((e + 112) << 23) | (m << 13);
    float r; std::memcpy(&r, &b, 4); return r;
}

// Round-trip a float through FP16 storage
static float fp16_rt(float x) {
    return fp16_bits_to_float(float_to_fp16_bits(x));
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  Precision Format Error Analysis
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> quantize_symmetric(const std::vector<float>& w,
                                              float scale, int q_max) {
    std::vector<float> out(w.size());
    for (int i = 0; i < (int)w.size(); i++) {
        int q = (int)std::round(w[i] / scale);
        q = std::max(-q_max, std::min(q_max, q));
        out[i] = q * scale;
    }
    return out;
}

static std::vector<float> quantize_block_symmetric(
        const std::vector<float>& w, int block_size, int bits) {
    int q_max = (1 << (bits - 1)) - 1;
    std::vector<float> out(w.size());
    for (int i = 0; i < (int)w.size(); i += block_size) {
        float max_a = 0;
        for (int k = i; k < i + block_size; k++)
            max_a = std::max(max_a, std::abs(w[k]));
        float scale = max_a > 0 ? max_a / q_max : 1.f;
        for (int k = i; k < i + block_size; k++) {
            int q = (int)std::round(w[k] / scale);
            q = std::max(-q_max, std::min(q_max, q));
            out[k] = q * scale;
        }
    }
    return out;
}

static void section1_precision_formats() {
    std::cout << std::string(70,'=') << "\n";
    std::cout << "§1  Precision Format Error Analysis\n";
    std::cout << std::string(70,'=') << "\n";

    const int n = 1024;
    auto w = make_weights(n, 0.02f);

    std::cout << "\n  " << n << " synthetic weights, N(0, 0.02^2)\n\n";
    std::cout << "  " << pad_right("Format",12) << "  " << pad_left("RMSE",12)
              << "  " << pad_left("MaxError",12)
              << "  " << pad_left("SNR(dB)",10)
              << "  " << pad_left("bpw",5) << "\n";
    std::cout << "  " << std::string(12,'-') << "  " << std::string(12,'-')
              << "  " << std::string(12,'-') << "  " << std::string(10,'-')
              << "  " << std::string(5,'-') << "\n";

    auto print_row = [&](const std::string& name,
                          const std::vector<float>& r,
                          const std::string& bpw) {
        std::cout << "  " << pad_right(name,12)
                  << "  " << pad_left(fmt(rmse(w,r),6),12)
                  << "  " << pad_left(fmt(max_abs_error(w,r),6),12)
                  << "  " << pad_left(fmt(snr_db(w,r),2),10)
                  << "  " << pad_left(bpw,5) << "\n";
    };

    // BF16
    std::vector<float> bf16(n);
    for (int i = 0; i < n; i++) bf16[i] = fp32_to_bf16_rt(w[i]);
    print_row("BF16", bf16, "16");

    // FP16
    std::vector<float> fp16(n);
    for (int i = 0; i < n; i++) fp16[i] = fp32_to_fp16_rt(w[i]);
    print_row("FP16", fp16, "16");

    // INT8 symmetric
    float max_a = *std::max_element(w.begin(), w.end(),
                   [](float a, float b){ return std::abs(a)<std::abs(b); });
    max_a = std::abs(max_a);
    auto int8 = quantize_symmetric(w, max_a / 127.f, 127);
    print_row("INT8 symm", int8, "8");

    // INT4 global
    auto int4g = quantize_symmetric(w, max_a / 7.f, 7);
    print_row("INT4 symm", int4g, "4");

    // INT4 block-wise (block=32)
    auto int4b = quantize_block_symmetric(w, 32, 4);
    print_row("INT4 b=32", int4b, "4.5");

    // INT2 block-wise
    auto int2b = quantize_block_symmetric(w, 32, 2);
    print_row("INT2 b=32", int2b, "2.5");

    // BF16 vs FP16 overflow
    std::cout << "\n  BF16 vs FP16 overflow test:\n";
    for (float v : {65000.f, 65504.f, 65505.f, 70000.f, 100000.f}) {
        float f16 = fp32_to_fp16_rt(v);
        float b16 = fp32_to_bf16_rt(v);
        std::cout << "    x=" << std::setw(10) << (int)v
                  << ":  FP16=" << std::setw(10) << f16
                  << "  BF16=" << std::setw(10) << b16 << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  GGUF Q8_0 Block Encoder / Decoder
// ─────────────────────────────────────────────────────────────────────────────

static const int BLOCK_Q8_0 = 32;

struct Q8_0Block {
    uint16_t scale_fp16;    // FP16 scale bits
    int8_t   quants[32];    // INT8 quantized weights

    static Q8_0Block from_floats(const float* src) {
        Q8_0Block b;
        float max_a = 0;
        for (int i = 0; i < 32; i++)
            max_a = std::max(max_a, std::abs(src[i]));
        float scale = max_a > 0 ? max_a / 127.f : 1.f;
        b.scale_fp16 = float_to_fp16_bits(scale);
        float sc = fp16_bits_to_float(b.scale_fp16);
        for (int i = 0; i < 32; i++) {
            int q = (int)std::round(src[i] / sc);
            b.quants[i] = (int8_t)std::max(-127, std::min(127, q));
        }
        return b;
    }

    void dequantize(float* dst) const {
        float sc = fp16_bits_to_float(scale_fp16);
        for (int i = 0; i < 32; i++)
            dst[i] = quants[i] * sc;
    }

    // Wire format: 2 bytes scale + 32 bytes quants = 34 bytes
    void encode(uint8_t* out) const {
        std::memcpy(out, &scale_fp16, 2);
        std::memcpy(out + 2, quants, 32);
    }
    static Q8_0Block decode(const uint8_t* in) {
        Q8_0Block b;
        std::memcpy(&b.scale_fp16, in, 2);
        std::memcpy(b.quants, in + 2, 32);
        return b;
    }
};

static void section2_q8_0() {
    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "§2  GGUF Q8_0: Block Encoder / Decoder\n";
    std::cout << std::string(70,'=') << "\n";

    const int n = 256;
    auto w = make_weights(n, 0.02f);
    int n_blocks = n / BLOCK_Q8_0;

    // Quantize
    std::vector<Q8_0Block> blocks(n_blocks);
    for (int b = 0; b < n_blocks; b++)
        blocks[b] = Q8_0Block::from_floats(w.data() + b * BLOCK_Q8_0);

    // Dequantize
    std::vector<float> recon(n);
    for (int b = 0; b < n_blocks; b++)
        blocks[b].dequantize(recon.data() + b * BLOCK_Q8_0);

    std::cout << "\n  n=" << n << " weights, " << n_blocks
              << " blocks of " << BLOCK_Q8_0 << "\n";
    std::cout << "  RMSE:      " << fmt(rmse(w,recon),6) << "\n";
    std::cout << "  Max error: " << fmt(max_abs_error(w,recon),6) << "\n";
    std::cout << "  SNR:       " << fmt(snr_db(w,recon),2) << " dB\n";

    // Wire round-trip
    std::vector<uint8_t> wire(n_blocks * 34);
    for (int b = 0; b < n_blocks; b++)
        blocks[b].encode(wire.data() + b * 34);
    std::vector<Q8_0Block> blocks2(n_blocks);
    for (int b = 0; b < n_blocks; b++)
        blocks2[b] = Q8_0Block::decode(wire.data() + b * 34);
    std::vector<float> recon2(n);
    for (int b = 0; b < n_blocks; b++)
        blocks2[b].dequantize(recon2.data() + b * BLOCK_Q8_0);

    std::cout << "\n  Wire round-trip (" << wire.size() << " bytes): "
              << "max error = " << fmt(max_abs_error(recon, recon2), 2) << "e+00\n";

    int fp16_bytes = n * 2;
    int q8_bytes   = (int)wire.size();
    std::cout << "\n  Memory:\n";
    std::cout << "    FP16:  " << fp16_bytes << " bytes\n";
    std::cout << "    Q8_0:  " << q8_bytes << " bytes  ("
              << fmt(100.0*q8_bytes/fp16_bytes,1) << "% of FP16, "
              << fmt((double)fp16_bytes/q8_bytes,2) << "x compression)\n";
    std::cout << "    bpw:   " << fmt(q8_bytes*8.0/n,2) << "\n";

    // Show first block
    auto& b0 = blocks[0];
    std::cout << "\n  Block 0 detail:\n";
    std::cout << "    scale (FP16): "
              << fmt(fp16_bits_to_float(b0.scale_fp16), 6) << "\n";
    std::cout << "    quants[0..7]: ";
    for (int i = 0; i < 8; i++) std::cout << (int)b0.quants[i] << " ";
    std::cout << "...\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  GGUF Q4_0 Block Encoder / Decoder (nibble packing)
// ─────────────────────────────────────────────────────────────────────────────

static const int BLOCK_Q4_0 = 32;

struct Q4_0Block {
    uint16_t scale_fp16;
    uint8_t  packed[16];    // 32 × 4-bit values, 2 per byte

    // unsigned nibble [0..15]; signed = nibble - 8
    static Q4_0Block from_floats(const float* src) {
        Q4_0Block b;
        float max_a = 0;
        for (int i = 0; i < 32; i++)
            max_a = std::max(max_a, std::abs(src[i]));
        float scale = max_a > 0 ? max_a / 7.f : 1.f;
        b.scale_fp16 = float_to_fp16_bits(scale);
        float sc = fp16_bits_to_float(b.scale_fp16);

        uint8_t nibbles[32];
        for (int i = 0; i < 32; i++) {
            int q = (int)std::round(src[i] / sc);
            q = std::max(-8, std::min(7, q));
            nibbles[i] = (uint8_t)(q + 8);    // [0..15]
        }
        for (int k = 0; k < 16; k++)
            b.packed[k] = (nibbles[2*k] & 0xF) | ((nibbles[2*k+1] & 0xF) << 4);
        return b;
    }

    void dequantize(float* dst) const {
        float sc = fp16_bits_to_float(scale_fp16);
        for (int k = 0; k < 16; k++) {
            dst[2*k]   = (int)( packed[k]       & 0xF) - 8) * sc;
            dst[2*k+1] = (int)((packed[k] >> 4) & 0xF) - 8) * sc;
        }
    }

    void encode(uint8_t* out) const {
        std::memcpy(out, &scale_fp16, 2);
        std::memcpy(out + 2, packed, 16);
    }
    static Q4_0Block decode(const uint8_t* in) {
        Q4_0Block b;
        std::memcpy(&b.scale_fp16, in, 2);
        std::memcpy(b.packed, in + 2, 16);
        return b;
    }
};

static void section3_q4_0() {
    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "§3  GGUF Q4_0: Nibble Packing Encoder / Decoder\n";
    std::cout << std::string(70,'=') << "\n";

    const int n = 256;
    auto w = make_weights(n, 0.02f);
    int n_blocks = n / BLOCK_Q4_0;

    std::vector<Q4_0Block> blocks(n_blocks);
    for (int b = 0; b < n_blocks; b++)
        blocks[b] = Q4_0Block::from_floats(w.data() + b * BLOCK_Q4_0);

    std::vector<float> recon(n);
    for (int b = 0; b < n_blocks; b++)
        blocks[b].dequantize(recon.data() + b * BLOCK_Q4_0);

    std::cout << "\n  n=" << n << " weights, " << n_blocks
              << " blocks of " << BLOCK_Q4_0 << "\n";
    std::cout << "  RMSE:      " << fmt(rmse(w,recon),6) << "\n";
    std::cout << "  Max error: " << fmt(max_abs_error(w,recon),6) << "\n";
    std::cout << "  SNR:       " << fmt(snr_db(w,recon),2) << " dB\n";

    // Wire round-trip
    std::vector<uint8_t> wire(n_blocks * 18);
    for (int b = 0; b < n_blocks; b++)
        blocks[b].encode(wire.data() + b * 18);
    std::vector<Q4_0Block> blocks2(n_blocks);
    for (int b = 0; b < n_blocks; b++)
        blocks2[b] = Q4_0Block::decode(wire.data() + b * 18);
    std::vector<float> recon2(n);
    for (int b = 0; b < n_blocks; b++)
        blocks2[b].dequantize(recon2.data() + b * BLOCK_Q4_0);

    std::cout << "\n  Wire round-trip (" << wire.size() << " bytes): "
              << "max error = " << fmt(max_abs_error(recon,recon2),2) << "e+00\n";

    int fp16_bytes = n * 2;
    int q4_bytes   = (int)wire.size();
    std::cout << "\n  Memory:\n";
    std::cout << "    FP16:  " << fp16_bytes << " bytes\n";
    std::cout << "    Q4_0:  " << q4_bytes << " bytes  ("
              << fmt(100.0*q4_bytes/fp16_bytes,1) << "% of FP16, "
              << fmt((double)fp16_bytes/q4_bytes,2) << "x compression)\n";
    std::cout << "    bpw:   " << fmt(q4_bytes*8.0/n,2) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  GGUF Q4_K Super-Block Encoder / Decoder
// ─────────────────────────────────────────────────────────────────────────────

static const int SUPER_BLOCK  = 256;
static const int INNER_BLOCK  =  32;
static const int N_INNER      =   8;   // SUPER_BLOCK / INNER_BLOCK
static const int SCALE_MAX_6  =  63;   // max 6-bit value

// Pack 8 × 6-bit values into 6 bytes
static void pack_6bit(const uint8_t* vals, uint8_t* out) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; i++)
        bits = (bits << 6) | (vals[i] & 0x3F);
    for (int i = 5; i >= 0; i--) {
        out[i] = (uint8_t)(bits & 0xFF);
        bits >>= 8;
    }
}

static void unpack_6bit(const uint8_t* in, uint8_t* vals) {
    uint64_t bits = 0;
    for (int i = 0; i < 6; i++)
        bits = (bits << 8) | in[i];
    for (int i = 7; i >= 0; i--) {
        vals[i] = (uint8_t)(bits & 0x3F);
        bits >>= 6;
    }
}

/*
 * Q4_K super-block (144 bytes):
 *   2B  d    = FP16 super_scale
 *   2B  dmin = FP16 super_min (positive; mins are SUBTRACTED during dequant)
 *   6B  scales[8]  (6-bit unsigned)
 *   6B  mins[8]    (6-bit unsigned magnitudes)
 *  128B  data (256 × 4-bit unsigned [0..15])
 *
 * Dequantize:  w = q × (scales[b] × d) − (mins[b] × dmin)
 */
struct Q4KSuperBlock {
    uint16_t d_fp16;      // super_scale
    uint16_t dmin_fp16;   // super_min  (positive)
    uint8_t  scales[8];   // 6-bit unsigned
    uint8_t  mins[8];     // 6-bit unsigned magnitudes
    uint8_t  data[128];   // packed nibbles (unsigned [0..15])

    static Q4KSuperBlock from_floats(const float* src) {
        Q4KSuperBlock sb;

        // Step 1: per-block asymmetric scale and min magnitude
        float block_scales[N_INNER], block_min_mags[N_INNER];
        for (int b = 0; b < N_INNER; b++) {
            const float* blk = src + b * INNER_BLOCK;
            float lo = blk[0], hi = blk[0];
            for (int i = 1; i < INNER_BLOCK; i++) {
                lo = std::min(lo, blk[i]);
                hi = std::max(hi, blk[i]);
            }
            float span = hi - lo;
            block_scales[b]   = span > 1e-12f ? span / 15.f : 1.f;
            block_min_mags[b] = std::abs(lo);
        }

        // Step 2: super-scale and super-min
        float max_sc = *std::max_element(block_scales, block_scales + N_INNER);
        float max_mn = *std::max_element(block_min_mags, block_min_mags + N_INNER);
        float d    = max_sc > 1e-12f ? max_sc / (float)SCALE_MAX_6 : 1.f;
        float dmin = max_mn > 1e-12f ? max_mn / (float)SCALE_MAX_6 : 1.f;
        sb.d_fp16    = float_to_fp16_bits(d);
        sb.dmin_fp16 = float_to_fp16_bits(dmin);
        float d_rt   = fp16_bits_to_float(sb.d_fp16);
        float dm_rt  = fp16_bits_to_float(sb.dmin_fp16);

        // Quantize scales and mins to 6-bit
        for (int b = 0; b < N_INNER; b++) {
            int sq = (int)std::round(block_scales[b]   / d_rt);
            int mq = (int)std::round(block_min_mags[b] / dm_rt);
            sb.scales[b] = (uint8_t)std::max(0, std::min(SCALE_MAX_6, sq));
            sb.mins[b]   = (uint8_t)std::max(0, std::min(SCALE_MAX_6, mq));
        }

        // Step 3: quantize weights
        uint8_t nibbles[SUPER_BLOCK];
        for (int b = 0; b < N_INNER; b++) {
            float rec_scale = sb.scales[b] * d_rt;
            float rec_min   = sb.mins[b]   * dm_rt;   // positive magnitude
            const float* blk = src + b * INNER_BLOCK;
            for (int i = 0; i < INNER_BLOCK; i++) {
                int q = 0;
                if (rec_scale > 1e-12f)
                    q = (int)std::round((blk[i] + rec_min) / rec_scale);
                nibbles[b * INNER_BLOCK + i] = (uint8_t)std::max(0, std::min(15, q));
            }
        }

        // Pack nibbles
        for (int k = 0; k < 128; k++)
            sb.data[k] = (nibbles[2*k] & 0xF) | ((nibbles[2*k+1] & 0xF) << 4);

        return sb;
    }

    void dequantize(float* dst) const {
        float d_rt  = fp16_bits_to_float(d_fp16);
        float dm_rt = fp16_bits_to_float(dmin_fp16);
        for (int b = 0; b < N_INNER; b++) {
            float rec_scale = scales[b] * d_rt;
            float rec_min   = mins[b]   * dm_rt;   // subtract this
            int start_byte  = b * 16;
            for (int k = 0; k < 16; k++) {
                uint8_t byte = data[start_byte + k];
                int lo = byte & 0xF;
                int hi = (byte >> 4) & 0xF;
                dst[b*INNER_BLOCK + 2*k]   = lo * rec_scale - rec_min;
                dst[b*INNER_BLOCK + 2*k+1] = hi * rec_scale - rec_min;
            }
        }
    }

    // Wire: 2+2+6+6+128 = 144 bytes
    void encode(uint8_t* out) const {
        std::memcpy(out, &d_fp16,    2); out += 2;
        std::memcpy(out, &dmin_fp16, 2); out += 2;
        pack_6bit(scales, out); out += 6;
        pack_6bit(mins,   out); out += 6;
        std::memcpy(out, data, 128);
    }

    static Q4KSuperBlock decode(const uint8_t* in) {
        Q4KSuperBlock sb;
        std::memcpy(&sb.d_fp16,    in, 2); in += 2;
        std::memcpy(&sb.dmin_fp16, in, 2); in += 2;
        unpack_6bit(in, sb.scales); in += 6;
        unpack_6bit(in, sb.mins);   in += 6;
        std::memcpy(sb.data, in, 128);
        return sb;
    }
};

static void section4_q4k() {
    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "§4  GGUF Q4_K: Super-Block Encoder / Decoder\n";
    std::cout << std::string(70,'=') << "\n";

    const int n = 512;   // 2 super-blocks
    auto w = make_weights(n, 0.02f);
    int n_sb = n / SUPER_BLOCK;

    std::vector<Q4KSuperBlock> sbs(n_sb);
    for (int b = 0; b < n_sb; b++)
        sbs[b] = Q4KSuperBlock::from_floats(w.data() + b * SUPER_BLOCK);

    std::vector<float> recon(n);
    for (int b = 0; b < n_sb; b++)
        sbs[b].dequantize(recon.data() + b * SUPER_BLOCK);

    std::cout << "\n  n=" << n << " weights, " << n_sb
              << " super-blocks of " << SUPER_BLOCK << "\n";
    std::cout << "  RMSE:      " << fmt(rmse(w,recon),6) << "\n";
    std::cout << "  Max error: " << fmt(max_abs_error(w,recon),6) << "\n";
    std::cout << "  SNR:       " << fmt(snr_db(w,recon),2) << " dB\n";

    // Wire round-trip
    std::vector<uint8_t> wire(n_sb * 144);
    for (int b = 0; b < n_sb; b++)
        sbs[b].encode(wire.data() + b * 144);
    assert((int)wire.size() == n_sb * 144);

    std::vector<Q4KSuperBlock> sbs2(n_sb);
    for (int b = 0; b < n_sb; b++)
        sbs2[b] = Q4KSuperBlock::decode(wire.data() + b * 144);
    std::vector<float> recon2(n);
    for (int b = 0; b < n_sb; b++)
        sbs2[b].dequantize(recon2.data() + b * SUPER_BLOCK);

    std::cout << "\n  Wire round-trip (" << wire.size() << " bytes): "
              << "max error = " << fmt(max_abs_error(recon,recon2),2) << "e+00\n";

    int fp16_bytes = n * 2;
    int q4k_bytes  = (int)wire.size();
    std::cout << "\n  Memory:\n";
    std::cout << "    FP16:   " << fp16_bytes << " bytes\n";
    std::cout << "    Q4_K:   " << q4k_bytes << " bytes  ("
              << fmt(100.0*q4k_bytes/fp16_bytes,1) << "% of FP16, "
              << fmt((double)fp16_bytes/q4k_bytes,2) << "x compression)\n";
    std::cout << "    bpw:    " << fmt(q4k_bytes*8.0/n,2) << "\n";

    // Super-block 0 detail
    auto& sb0 = sbs[0];
    float d0   = fp16_bits_to_float(sb0.d_fp16);
    float dm0  = fp16_bits_to_float(sb0.dmin_fp16);
    std::cout << "\n  Super-block 0 detail:\n";
    std::cout << "    super_scale: " << fmt(d0,6) << "\n";
    std::cout << "    super_min:   " << fmt(dm0,6) << "\n";
    std::cout << "    scales (6-bit): ";
    for (int i = 0; i < 8; i++) std::cout << (int)sb0.scales[i] << " ";
    std::cout << "\n";
    std::cout << "    mins   (6-bit): ";
    for (int i = 0; i < 8; i++) std::cout << (int)sb0.mins[i]   << " ";
    std::cout << "\n";

    // Compare Q4_0 vs Q4_K
    int n_q4_blocks = n / BLOCK_Q4_0;
    std::vector<Q4_0Block> q40_blocks(n_q4_blocks);
    for (int b = 0; b < n_q4_blocks; b++)
        q40_blocks[b] = Q4_0Block::from_floats(w.data() + b * BLOCK_Q4_0);
    std::vector<float> recon_q40(n);
    for (int b = 0; b < n_q4_blocks; b++)
        q40_blocks[b].dequantize(recon_q40.data() + b * BLOCK_Q4_0);

    std::cout << "\n  Q4_0 vs Q4_K accuracy (same " << n << " weights):\n";
    std::cout << "    Q4_0 RMSE: " << fmt(rmse(w,recon_q40),6)
              << "  SNR: " << fmt(snr_db(w,recon_q40),2) << " dB\n";
    std::cout << "    Q4_K RMSE: " << fmt(rmse(w,recon),6)
              << "  SNR: " << fmt(snr_db(w,recon),2) << " dB\n";
    double adv = snr_db(w,recon) - snr_db(w,recon_q40);
    std::cout << "    Q4_K advantage: " << (adv >= 0 ? "+" : "") << fmt(adv,2)
              << " dB SNR\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  GPTQ-Style Error Propagation (simplified)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> rtn_row(const std::vector<float>& row,
                                   float scale, int q_max) {
    std::vector<float> out(row.size());
    for (int i = 0; i < (int)row.size(); i++) {
        int q = (int)std::round(row[i] / scale);
        q = std::max(-q_max, std::min(q_max, q));
        out[i] = q * scale;
    }
    return out;
}

static std::vector<float> gptq_row(const std::vector<float>& row,
                                    float scale, int q_max) {
    int n = (int)row.size();
    std::vector<float> work(row);
    std::vector<float> out(n);
    for (int j = 0; j < n; j++) {
        int q = (int)std::round(work[j] / scale);
        q = std::max(-q_max, std::min(q_max, q));
        out[j]   = q * scale;
        float err = work[j] - out[j];
        if (j + 1 < n) {
            float corr = err / (n - j - 1);
            for (int k = j + 1; k < n; k++)
                work[k] += corr;
        }
    }
    return out;
}

static void section5_gptq() {
    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "§5  GPTQ-Style Error Propagation (simplified)\n";
    std::cout << std::string(70,'=') << "\n";

    const int n = 256;
    auto w = make_weights(n, 0.02f);

    float max_a = 0;
    for (float v : w) max_a = std::max(max_a, std::abs(v));
    float scale = max_a / 7.f;

    auto rtn  = rtn_row(w, scale, 7);
    auto gptq = gptq_row(w, scale, 7);

    std::cout << "\n  n=" << n << " weights, INT4, scale=" << fmt(scale,6) << "\n\n";
    std::cout << "  " << pad_right("Method",20) << "  " << pad_left("RMSE",12)
              << "  " << pad_left("SNR(dB)",10) << "\n";
    std::cout << "  " << std::string(20,'-') << "  " << std::string(12,'-')
              << "  " << std::string(10,'-') << "\n";
    std::cout << "  " << pad_right("RTN",20)
              << "  " << pad_left(fmt(rmse(w,rtn),6),12)
              << "  " << pad_left(fmt(snr_db(w,rtn),2),10) << "\n";
    std::cout << "  " << pad_right("GPTQ (err.prop.)",20)
              << "  " << pad_left(fmt(rmse(w,gptq),6),12)
              << "  " << pad_left(fmt(snr_db(w,gptq),2),10) << "\n";

    double adv = snr_db(w,gptq) - snr_db(w,rtn);
    std::cout << "\n  GPTQ advantage: " << (adv >= 0 ? "+" : "") << fmt(adv,2) << " dB\n";
    std::cout << "  (Real GPTQ uses Hessian weighting; uniform here)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  AWQ Per-Channel Activation-Aware Scaling
// ─────────────────────────────────────────────────────────────────────────────

static void section6_awq() {
    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "§6  AWQ: Activation-Aware Per-Channel Scaling\n";
    std::cout << std::string(70,'=') << "\n";

    const int n_rows = 16, n_cols = 64;
    Lcg rng(42);

    // Weight matrix
    std::vector<std::vector<float>> W(n_rows, std::vector<float>(n_cols));
    for (auto& row : W)
        for (auto& v : row) v = rng.next_normal(0.f, 0.02f);

    // Activation scales with a few salient channels
    std::vector<float> act(n_cols);
    for (auto& v : act) v = rng.next_uniform() * 0.09f + 0.01f;
    // Make channels 3, 19, 47 salient
    std::vector<int> salient = {3, 19, 47};
    for (int c : salient) act[c] *= 20.f;

    // AWQ scale: s[c] = act[c]^0.5
    std::vector<float> s(n_cols);
    for (int c = 0; c < n_cols; c++) s[c] = std::sqrt(act[c]);

    auto quantize_row_int4 = [&](const std::vector<float>& row)
            -> std::vector<float> {
        float max_a = 0;
        for (float v : row) max_a = std::max(max_a, std::abs(v));
        float scale = max_a > 0 ? max_a / 7.f : 1.f;
        std::vector<float> out(row.size());
        for (int i = 0; i < (int)row.size(); i++) {
            int q = (int)std::round(row[i] / scale);
            q = std::max(-7, std::min(7, q));
            out[i] = q * scale;
        }
        return out;
    };

    // RTN
    double rtn_rmse = 0, rtn_wrmse = 0;
    for (int r = 0; r < n_rows; r++) {
        auto dq = quantize_row_int4(W[r]);
        for (int c = 0; c < n_cols; c++) {
            double d = W[r][c] - dq[c];
            rtn_rmse  += d*d;
            rtn_wrmse += (act[c] * d) * (act[c] * d);
        }
    }
    rtn_rmse  = std::sqrt(rtn_rmse  / (n_rows * n_cols));
    rtn_wrmse = std::sqrt(rtn_wrmse / (n_rows * n_cols));

    // AWQ
    double awq_rmse = 0, awq_wrmse = 0;
    for (int r = 0; r < n_rows; r++) {
        // Scale weights
        std::vector<float> scaled(n_cols);
        for (int c = 0; c < n_cols; c++) scaled[c] = W[r][c] * s[c];
        // Quantize scaled
        auto dq_scaled = quantize_row_int4(scaled);
        // Undo scaling and compute error
        for (int c = 0; c < n_cols; c++) {
            float dq = dq_scaled[c] / s[c];
            double d = W[r][c] - dq;
            awq_rmse  += d*d;
            awq_wrmse += (act[c] * d) * (act[c] * d);
        }
    }
    awq_rmse  = std::sqrt(awq_rmse  / (n_rows * n_cols));
    awq_wrmse = std::sqrt(awq_wrmse / (n_rows * n_cols));

    std::cout << "\n  Weight matrix: " << n_rows << "x" << n_cols
              << ", INT4, salient channels: 3, 19, 47\n\n";
    std::cout << "  " << pad_right("Method",15) << "  " << pad_left("RMSE",10)
              << "  " << pad_left("Weighted RMSE",15) << "\n";
    std::cout << "  " << std::string(15,'-') << "  " << std::string(10,'-')
              << "  " << std::string(15,'-') << "\n";
    std::cout << "  " << pad_right("RTN",15)
              << "  " << pad_left(fmt(rtn_rmse,6),10)
              << "  " << pad_left(fmt(rtn_wrmse,6),15) << "\n";
    std::cout << "  " << pad_right("AWQ",15)
              << "  " << pad_left(fmt(awq_rmse,6),10)
              << "  " << pad_left(fmt(awq_wrmse,6),15) << "\n";
    double impr = 100.0 * (1.0 - awq_wrmse / rtn_wrmse);
    std::cout << "\n  AWQ improves weighted RMSE by " << fmt(impr,1) << "%\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  KV Cache INT8 Quantization and Attention Accuracy
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> softmax_vec(const std::vector<float>& x) {
    float mx = *std::max_element(x.begin(), x.end());
    std::vector<float> e(x.size());
    float s = 0;
    for (int i = 0; i < (int)x.size(); i++) { e[i] = std::exp(x[i] - mx); s += e[i]; }
    for (auto& v : e) v /= s;
    return e;
}

static std::vector<float> attn_scores(
        const std::vector<float>& Q,
        const std::vector<std::vector<float>>& K, int d_head) {
    float scale = 1.f / std::sqrt((float)d_head);
    std::vector<float> scores(K.size());
    for (int t = 0; t < (int)K.size(); t++) {
        float dot = 0;
        for (int i = 0; i < d_head; i++) dot += Q[i] * K[t][i];
        scores[t] = dot * scale;
    }
    return softmax_vec(scores);
}

static void section7_kv_quant() {
    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "§7  KV Cache INT8 Quantization and Attention Accuracy\n";
    std::cout << std::string(70,'=') << "\n";

    const int seq_len = 64, d_head = 128;
    Lcg rng(42);

    auto rand_vec = [&](int d) {
        std::vector<float> v(d);
        for (auto& x : v) x = rng.next_normal(0.f, 0.1f);
        return v;
    };

    std::vector<std::vector<float>> K(seq_len), V(seq_len);
    for (int t = 0; t < seq_len; t++) { K[t] = rand_vec(d_head); V[t] = rand_vec(d_head); }
    auto Q = rand_vec(d_head);

    // Reference
    auto scores_ref = attn_scores(Q, K, d_head);
    std::vector<float> out_ref(d_head, 0);
    for (int t = 0; t < seq_len; t++)
        for (int i = 0; i < d_head; i++) out_ref[i] += scores_ref[t] * V[t][i];

    // Quantize KV to INT8
    auto quant_kv = [&](const std::vector<std::vector<float>>& vecs, int bits)
            -> std::vector<std::vector<float>> {
        int q_max = (1 << (bits - 1)) - 1;
        std::vector<std::vector<float>> dq(vecs.size());
        for (int t = 0; t < (int)vecs.size(); t++) {
            float max_a = 0;
            for (float v : vecs[t]) max_a = std::max(max_a, std::abs(v));
            float sc = max_a > 0 ? max_a / q_max : 1.f;
            dq[t].resize(d_head);
            for (int i = 0; i < d_head; i++) {
                int q = (int)std::round(vecs[t][i] / sc);
                q = std::max(-q_max, std::min(q_max, q));
                dq[t][i] = q * sc;
            }
        }
        return dq;
    };

    auto eval = [&](int bits, const std::string& label) {
        auto K_dq = quant_kv(K, bits);
        auto V_dq = quant_kv(V, bits);
        auto scores = attn_scores(Q, K_dq, d_head);
        std::vector<float> out(d_head, 0);
        for (int t = 0; t < seq_len; t++)
            for (int i = 0; i < d_head; i++) out[i] += scores[t] * V_dq[t][i];

        double kl = 0;
        for (int t = 0; t < seq_len; t++)
            if (scores_ref[t] > 0)
                kl += scores_ref[t] * std::log(scores_ref[t] / (scores[t] + 1e-12) + 1e-12);

        std::cout << "  " << pad_right(label,12)
                  << "  " << pad_left(fmt(rmse(out_ref, out),6),12)
                  << "  " << pad_left(fmt(kl,8),14)
                  << "  " << pad_left(std::to_string(bits),5) << "\n";
    };

    std::cout << "\n  seq_len=" << seq_len << ", d_head=" << d_head << "\n\n";
    std::cout << "  " << pad_right("Format",12) << "  " << pad_left("Output RMSE",12)
              << "  " << pad_left("Score KL-div",14) << "  " << pad_left("bpw",5) << "\n";
    std::cout << "  " << std::string(12,'-') << "  " << std::string(12,'-')
              << "  " << std::string(14,'-') << "  " << std::string(5,'-') << "\n";
    std::cout << "  " << pad_right("FP32 (ref)",12)
              << "  " << pad_left("0.000000",12) << "  " << pad_left("0.00000000",14)
              << "  " << pad_left("32",5) << "\n";
    eval(8, "INT8 KV");
    eval(4, "INT4 KV");

    int fp16_kv = seq_len * d_head * 2 * 2;
    int int8_kv = seq_len * d_head * 1 * 2;
    int int4_kv = seq_len * d_head / 2 * 2;
    std::cout << "\n  KV memory (seq=" << seq_len << ", d_head=" << d_head << "):\n";
    std::cout << "    BF16: " << fp16_kv << " bytes  (1.0x)\n";
    std::cout << "    INT8: " << int8_kv << " bytes  ("
              << fmt((double)fp16_kv/int8_kv,1) << "x reduction)\n";
    std::cout << "    INT4: " << int4_kv << " bytes  ("
              << fmt((double)fp16_kv/int4_kv,1) << "x reduction)\n";
    std::cout << "\n  [COMMON TRAP] INT4 KV degrades quality at long contexts."
                 " Use INT8 as floor.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  Roofline Speedup Predictions
// ─────────────────────────────────────────────────────────────────────────────

struct HWSpec {
    const char* name;
    double peak_tflops;
    double hbm_bw_gbs;
};
struct QScheme {
    const char* name;
    double bpw;
    double dequant_flops_per_w;
    double practical_frac;
};

static const HWSpec HW[] = {
    {"A100 80GB",  312.0,  2000.0},
    {"H100 SXM5", 1979.0,  3350.0},
    {"RTX 4090",   330.0,  1008.0},
};
static const QScheme SCHEMES[] = {
    {"BF16 (baseline)", 16.0, 0.0,  1.00},
    {"FP8 (H100)",       8.0, 0.1,  0.85},
    {"Q8_0",             8.5, 0.5,  0.80},
    {"Q6_K",             6.6, 1.0,  0.75},
    {"Q5_K_M",           5.7, 1.5,  0.72},
    {"Q4_K_M",           4.5, 2.1,  0.70},
    {"Q3_K_M",           3.4, 3.0,  0.65},
    {"Q2_K",             2.6, 4.0,  0.60},
};

static void section8_roofline() {
    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "§8  Roofline Speedup Predictions per Quantization Scheme\n";
    std::cout << std::string(70,'=') << "\n";

    const double n_weights = 7.03e9;   // LLaMA 3 8B non-embedding

    for (const auto& hw : HW) {
        std::cout << "\n  Hardware: " << hw.name
                  << "  (peak " << fmt(hw.peak_tflops,0) << " TFLOP/s, "
                  << "HBM " << fmt(hw.hbm_bw_gbs,0) << " GB/s)\n";
        std::cout << "  Model: LLaMA 3 8B, B=1 (decode)\n\n";
        std::cout << "  " << pad_right("Scheme",18) << "  " << pad_left("bpw",5)
                  << "  " << pad_left("BW time",9) << "  " << pad_left("Dequant",9)
                  << "  " << pad_left("Theory",8) << "  " << pad_left("Practical",10) << "\n";
        std::cout << "  " << std::string(18,'-') << "  " << std::string(5,'-')
                  << "  " << std::string(9,'-') << "  " << std::string(9,'-')
                  << "  " << std::string(8,'-') << "  " << std::string(10,'-') << "\n";

        double bf16_ms = (n_weights * 2.0) / (hw.hbm_bw_gbs * 1e9) * 1000.0;

        for (const auto& s : SCHEMES) {
            double bw_time_ms = (n_weights * s.bpw / 8.0)
                                / (hw.hbm_bw_gbs * 1e9) * 1000.0;
            double dq_flops   = n_weights * s.dequant_flops_per_w;
            double dq_time_ms = dq_flops / (hw.peak_tflops * 1e12) * 1000.0;
            double quant_ms   = bw_time_ms + dq_time_ms;
            double theory     = bf16_ms / quant_ms;
            double practical  = theory * s.practical_frac;

            std::cout << "  " << pad_right(s.name,18)
                      << "  " << pad_left(fmt(s.bpw,1),5)
                      << "  " << pad_left(fmt(bw_time_ms,2)+"ms",9)
                      << "  " << pad_left(fmt(dq_time_ms*1000,2)+"us",9)
                      << "  " << pad_left(fmt(theory,2)+"x",8)
                      << "  " << pad_left(fmt(practical,2)+"x",10) << "\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    section1_precision_formats();
    section2_q8_0();
    section3_q4_0();
    section4_q4k();
    section5_gptq();
    section6_awq();
    section7_kv_quant();
    section8_roofline();
    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "All sections complete.\n";
    std::cout << std::string(70,'=') << "\n";
    return 0;
}

```

