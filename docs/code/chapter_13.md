# Chapter 13: Token Streaming — The Last Mile — Companion Code

## Python — `streaming_demo.py`

```python
"""
Chapter 13 — Token Streaming: The Last Mile
Companion code: streaming_demo.py

Demonstrates:
  1. UTF-8 boundary detector and streaming detokenizer
  2. SSE frame parser (handles multi-line events, [DONE] sentinel)
  3. TTFT and ITL measurement harness over a simulated stream
  4. Simulated vLLM streaming response with realistic timing
  5. Backpressure simulation: slow consumer vs. fast producer
  6. End-to-end latency breakdown visualiser

No GPU required — all operations are CPU/IO simulation.
"""

import time
import json
import math
import random
import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — UTF-8 boundary detector and streaming detokenizer
# ─────────────────────────────────────────────────────────────────────────────

def utf8_leading_byte_length(byte: int) -> int:
    """
    Given a leading byte, return the total number of bytes in the UTF-8
    sequence it starts. Returns -1 for continuation bytes (invalid leaders).
    """
    if (byte & 0x80) == 0x00:  return 1   # 0xxxxxxx — ASCII
    if (byte & 0xE0) == 0xC0:  return 2   # 110xxxxx — 2-byte
    if (byte & 0xF0) == 0xE0:  return 3   # 1110xxxx — 3-byte
    if (byte & 0xF8) == 0xF0:  return 4   # 11110xxx — 4-byte
    return -1                              # 10xxxxxx — continuation byte


class StreamingDetokenizer:
    """
    Accumulates raw bytes from the token stream and emits complete
    UTF-8 characters only — never partial sequences.
    """

    def __init__(self, suppress_special: bool = True):
        self._buffer: bytearray = bytearray()
        self._expected_len: int = 0   # expected total bytes for current char
        self.suppress_special = suppress_special
        # Simplified special token list (real vocab has hundreds)
        self._special_strings = {
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
        }

    def feed(self, raw_bytes: bytes) -> str:
        """
        Feed raw bytes from one token. Returns any complete UTF-8 text
        ready to emit. May return empty string if bytes are partial.
        """
        emitted = []

        for byte in raw_bytes:
            if len(self._buffer) == 0:
                # Start of a new character
                seq_len = utf8_leading_byte_length(byte)
                if seq_len == -1:
                    # Continuation byte without leader — skip (encoding error)
                    continue
                self._expected_len = seq_len
                self._buffer.append(byte)
            else:
                # Continuation byte
                self._buffer.append(byte)

            # Check if the current character is complete
            if len(self._buffer) == self._expected_len:
                try:
                    char = self._buffer.decode("utf-8")
                    if not self.suppress_special or char not in self._special_strings:
                        emitted.append(char)
                except UnicodeDecodeError:
                    pass  # malformed — skip
                self._buffer.clear()
                self._expected_len = 0

        return "".join(emitted)

    def flush(self) -> str:
        """Emit any remaining buffered bytes (force-flush at EOS)."""
        if self._buffer:
            try:
                return self._buffer.decode("utf-8", errors="replace")
            finally:
                self._buffer.clear()
        return ""


def demo_utf8_detokenizer():
    print("=" * 65)
    print("SECTION 1: UTF-8 Streaming Detokenizer")
    print("=" * 65)
    print()

    # Test cases: bytes split across token boundaries
    test_cases = [
        ("ASCII safe",        [b"The ", b"cat ", b"sat."]),
        ("2-byte char split", [b"caf\xC3", b"\xA9"]),          # "café" split
        ("3-byte char split", [b"\xE4\xB8", b"\xAD\xE6\x96"   # Chinese chars split
                               , b"\x87"]),
        ("Special token",     [b"hello", b"<|eot_id|>", b"!"]),
        ("Emoji (4-byte)",    [b"\xF0\x9F\x98", b"\x80"]),      # 😀 split
    ]

    for label, token_bytes_list in test_cases:
        det = StreamingDetokenizer()
        parts = []
        for tb in token_bytes_list:
            chunk = det.feed(tb)
            parts.append(repr(chunk) if chunk else "(buffered)")
        flush = det.flush()
        if flush:
            parts.append(repr(flush) + " [flush]")

        full = "".join(p for p in parts if p != "(buffered)")
        print(f"  {label:<25} tokens → {[repr(b) for b in token_bytes_list]}")
        print(f"  {'':25} emitted per token: {parts}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — SSE frame parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_sse_stream(raw_text: str) -> Iterator[dict]:
    """
    Parse a raw SSE response string into event dicts.
    Handles multi-line data, id:, event: fields, and [DONE] sentinel.
    """
    event_lines = []
    for line in raw_text.splitlines():
        if line == "":
            # Empty line = end of event block
            if event_lines:
                event = {}
                data_parts = []
                for l in event_lines:
                    if l.startswith("data: "):
                        data_parts.append(l[6:])
                    elif l.startswith("id: "):
                        event["id"] = l[4:]
                    elif l.startswith("event: "):
                        event["event"] = l[7:]
                if data_parts:
                    raw_data = "\n".join(data_parts)
                    event["data"] = raw_data
                    if raw_data == "[DONE]":
                        event["done"] = True
                    else:
                        try:
                            event["parsed"] = json.loads(raw_data)
                        except json.JSONDecodeError:
                            event["raw"] = raw_data
                yield event
                event_lines = []
        else:
            event_lines.append(line)


def build_sse_frame(content: str, request_id: str, finish_reason: Optional[str] = None) -> str:
    """Build one OpenAI-compatible SSE frame for a token delta."""
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


def demo_sse_parser():
    print("=" * 65)
    print("SECTION 2: SSE Frame Parser")
    print("=" * 65)
    print()

    # Simulate a multi-token SSE response
    tokens = ["The", " quick", " brown", " fox", " jumps", "."]
    raw_sse = ""
    for i, tok in enumerate(tokens):
        is_last = i == len(tokens) - 1
        raw_sse += build_sse_frame(tok, "req-demo", "stop" if is_last else None)
    raw_sse += "data: [DONE]\n\n"

    print("  Raw SSE bytes (first 200 chars):")
    print("  " + repr(raw_sse[:200]))
    print()

    print("  Parsed events:")
    reconstructed = ""
    for evt in parse_sse_stream(raw_sse):
        if evt.get("done"):
            print("  [DONE] sentinel received")
            break
        parsed = evt.get("parsed", {})
        choices = parsed.get("choices", [{}])
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        finish = choices[0].get("finish_reason")
        reconstructed += content
        print(f"  token={repr(content):<12} finish_reason={finish}")

    print(f"\n  Reconstructed text: {repr(reconstructed)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Simulated vLLM streaming response with timing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulatedToken:
    token_text: str
    gpu_time_ms: float    # forward pass latency
    sample_time_ms: float # sampling latency
    detok_time_ms: float  # detokenization latency
    send_time_ms: float   # SSE frame + socket write


def simulate_vllm_stream(
    tokens: list[str],
    prefill_tokens: int = 200,
    prefill_speed: float = 40000.0,    # tokens/sec
    decode_itl_ms: float = 18.0,       # ms per decode step (GPU)
    network_ms: float = 0.5,           # ms per token (LAN)
    jitter_pct: float = 0.10,          # ±10% timing jitter
) -> list[SimulatedToken]:
    """
    Simulate the per-token timing of a vLLM streaming response.
    Returns a list of SimulatedToken with per-stage breakdowns.
    """
    rng = random.Random(42)

    def jitter(base_ms: float) -> float:
        return base_ms * (1.0 + rng.uniform(-jitter_pct, jitter_pct))

    result = []
    for tok in tokens:
        result.append(SimulatedToken(
            token_text=tok,
            gpu_time_ms=jitter(decode_itl_ms),
            sample_time_ms=jitter(0.10),
            detok_time_ms=jitter(0.05),
            send_time_ms=jitter(0.35) + jitter(network_ms),
        ))
    return result


def demo_timing_breakdown():
    print("=" * 65)
    print("SECTION 3: End-to-End Latency Breakdown")
    print("=" * 65)
    print()

    tokens = ["The", " quick", " brown", " fox", " jumps", " over",
              " the", " lazy", " dog", "."]

    # LAN vs WAN comparison
    for label, network_ms, decode_ms in [
        ("LAN (same DC, 0.5ms RTT)", 0.5, 18.0),
        ("WAN (cross-region, 80ms RTT)", 80.0, 18.0),
    ]:
        stream = simulate_vllm_stream(tokens, network_ms=network_ms,
                                       decode_itl_ms=decode_ms)
        print(f"  Scenario: {label}")
        print(f"  {'Token':<10} {'GPU (ms)':>9} {'Sample':>7} {'Detok':>7} "
              f"{'Send':>8} {'Total':>8}")
        print("  " + "-" * 55)

        totals = [0.0, 0.0, 0.0, 0.0]
        for st in stream:
            total = (st.gpu_time_ms + st.sample_time_ms +
                     st.detok_time_ms + st.send_time_ms)
            print(f"  {st.token_text:<10} {st.gpu_time_ms:>9.1f} "
                  f"{st.sample_time_ms:>7.2f} {st.detok_time_ms:>7.2f} "
                  f"{st.send_time_ms:>8.1f} {total:>8.1f}")
            totals[0] += st.gpu_time_ms
            totals[1] += st.sample_time_ms
            totals[2] += st.detok_time_ms
            totals[3] += st.send_time_ms

        total_all = sum(totals)
        avg = total_all / len(stream)
        print(f"  {'AVERAGE':<10} {totals[0]/len(stream):>9.1f} "
              f"{totals[1]/len(stream):>7.2f} {totals[2]/len(stream):>7.2f} "
              f"{totals[3]/len(stream):>8.1f} {avg:>8.1f}")
        print(f"\n  Effective rate: {1000/avg:.1f} tokens/sec")
        print(f"  GPU share of latency: {totals[0]/total_all*100:.0f}%")
        print(f"  Network share:        {totals[3]/total_all*100:.0f}%\n")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — TTFT and ITL measurement harness
# ─────────────────────────────────────────────────────────────────────────────

async def async_token_generator(
    tokens: list[str],
    prefill_ms: float = 120.0,
    itl_ms: float = 18.0,
) -> AsyncIterator[str]:
    """Simulate an async vLLM token generator with realistic timing."""
    await asyncio.sleep(prefill_ms / 1000.0)   # prefill delay
    for tok in tokens:
        yield tok
        await asyncio.sleep(itl_ms / 1000.0)  # decode step delay


async def measure_streaming_metrics(
    tokens: list[str],
    prefill_ms: float = 120.0,
    itl_ms: float = 18.0,
) -> dict:
    """Measure TTFT and ITL statistics from a simulated stream."""
    t_start = time.perf_counter()
    first_token_t = None
    token_times = []

    async for tok in async_token_generator(tokens, prefill_ms, itl_ms):
        t_now = time.perf_counter()
        if first_token_t is None:
            first_token_t = t_now
        token_times.append(t_now)

    ttft_ms = (first_token_t - t_start) * 1000 if first_token_t else 0.0
    itls = [(token_times[i] - token_times[i-1]) * 1000
            for i in range(1, len(token_times))]

    def percentile(data, p):
        if not data:
            return 0.0
        idx = int(len(data) * p / 100)
        return sorted(data)[min(idx, len(data)-1)]

    return {
        "ttft_ms": ttft_ms,
        "n_tokens": len(tokens),
        "itl_p50": percentile(itls, 50),
        "itl_p95": percentile(itls, 95),
        "itl_p99": percentile(itls, 99),
        "throughput_tok_s": len(tokens) / ((time.perf_counter() - t_start)),
    }


def demo_ttft_measurement():
    print("=" * 65)
    print("SECTION 4: TTFT and ITL Measurement Harness")
    print("=" * 65)
    print()

    tokens = ["The", " quick", " brown", " fox", " jumps", " over",
              " the", " lazy", " dog", "."] * 5   # 50 tokens

    scenarios = [
        ("Short prompt (500T)",   50.0,  18.0),
        ("RAG prompt (8K T)",    200.0,  18.0),
        ("Long prompt (32K T)",  800.0,  18.0),
    ]

    for label, prefill_ms, itl_ms in scenarios:
        metrics = asyncio.run(measure_streaming_metrics(tokens, prefill_ms, itl_ms))
        print(f"  [{label}]")
        print(f"    TTFT:          {metrics['ttft_ms']:>7.1f} ms")
        print(f"    ITL p50:       {metrics['itl_p50']:>7.1f} ms")
        print(f"    ITL p95:       {metrics['itl_p95']:>7.1f} ms")
        print(f"    Throughput:    {metrics['throughput_tok_s']:>7.1f} tok/s")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Backpressure simulation
# ─────────────────────────────────────────────────────────────────────────────

def demo_backpressure():
    print("=" * 65)
    print("SECTION 5: Backpressure Simulation")
    print("=" * 65)
    print()

    print("  Producer: generates 1 token every 18 ms (GPU decode)")
    print("  Scenario A: Fast consumer (reads every 10 ms) — buffer stays small")
    print("  Scenario B: Slow consumer (reads every 50 ms) — buffer grows")
    print()

    def simulate(producer_ms: float, consumer_ms: float,
                 n_tokens: int = 20) -> dict:
        buffer: deque = deque()
        buffer_sizes = []
        t_produce = 0.0
        t_consume = 0.0

        produced = 0
        consumed = 0
        t_now = 0.0

        while consumed < n_tokens:
            # Produce tokens up to current time
            while produced < n_tokens and t_produce <= t_now:
                buffer.append(produced)
                produced += 1
                t_produce += producer_ms

            # Consume one token if available and consumer is ready
            if buffer and t_consume <= t_now:
                buffer.popleft()
                consumed += 1
                t_consume += consumer_ms

            buffer_sizes.append(len(buffer))
            t_now += 1.0   # 1ms tick

        return {
            "max_buffer": max(buffer_sizes),
            "avg_buffer": sum(buffer_sizes) / len(buffer_sizes),
            "finish_time_ms": t_now,
        }

    for label, prod_ms, cons_ms in [
        ("Fast consumer (10 ms read)", 18.0, 10.0),
        ("Matched consumer (18 ms)",   18.0, 18.0),
        ("Slow consumer (50 ms read)", 18.0, 50.0),
        ("Very slow (200 ms read)",    18.0, 200.0),
    ]:
        r = simulate(prod_ms, cons_ms)
        print(f"  {label:<35}  max_buf={r['max_buffer']:>3}  "
              f"avg_buf={r['avg_buffer']:.1f}  "
              f"finish={r['finish_time_ms']:.0f} ms")

    print()
    print("  Implication for vLLM:")
    print("    Slow consumers accumulate tokens in the output queue.")
    print("    vLLM backpressure eventually throttles scheduling for slow requests.")
    print("    engine.abort(request_id) frees KV blocks when client disconnects.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — End-to-end latency breakdown visualiser
# ─────────────────────────────────────────────────────────────────────────────

def demo_latency_breakdown_visual():
    print("=" * 65)
    print("SECTION 6: Latency Breakdown Visual (LAN vs WAN)")
    print("=" * 65)
    print()

    components = [
        ("GPU forward pass",   18.0,  18.0),
        ("Sampling (CPU)",      0.1,   0.1),
        ("Detokenization",      0.05,  0.05),
        ("SSE frame build",     0.05,  0.05),
        ("Kernel write+TCP",    0.3,   0.3),
        ("Network transit",     0.5,  80.0),
        ("Browser render",      1.0,   1.0),
    ]

    for scenario, net_idx in [("LAN", 0), ("WAN", 1)]:
        total = sum(c[net_idx + 1] for c in components)
        print(f"  {scenario} — Total per token: {total:.2f} ms")
        print(f"  {'Component':<25} {'Time (ms)':>10}  {'Share':>6}  Bar")
        print("  " + "-" * 60)
        for name, lan_ms, wan_ms in components:
            ms = lan_ms if net_idx == 0 else wan_ms
            pct = ms / total * 100
            bar = "█" * max(1, int(pct / 2))
            print(f"  {name:<25} {ms:>10.2f}  {pct:>5.1f}%  {bar}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_utf8_detokenizer()
    demo_sse_parser()
    demo_timing_breakdown()
    demo_ttft_measurement()
    demo_backpressure()
    demo_latency_breakdown_visual()

    print("=" * 65)
    print("Chapter 13 streaming demo complete.")
    print()
    print("Key results:")
    print("  • UTF-8 boundary handling: buffer incomplete sequences,")
    print("    emit only on complete character assembly")
    print("  • SSE frame: 'data: {...}\\n\\n' per token, '[DONE]' sentinel")
    print("  • LAN: GPU dominates (90%+ of latency)")
    print("  • WAN: network dominates (~80%+ at 80ms RTT)")
    print("  • Slow consumers: buffer grows; abort() frees KV blocks")
    print("=" * 65)

```

## C++ — `streaming_demo.cpp`

```cpp
/*
 * Chapter 13 — Token Streaming: The Last Mile
 * Companion code: streaming_demo.cpp
 *
 * Demonstrates:
 *   1. UTF-8 boundary check (utf8_continuation_bytes)
 *   2. Streaming detokenizer with boundary-safe byte accumulation
 *   3. SSE frame builder: JSON encode + "data: ...\n\n" wrapper
 *   4. Token-by-token decode loop (annotated llama.cpp pattern)
 *   5. Chunked HTTP response simulator
 *   6. End-to-end latency breakdown timer
 *   7. Backpressure ring buffer simulation
 *
 * Compile:
 *   g++ -std=c++17 -O2 -o streaming_demo streaming_demo.cpp
 *
 * Run:
 *   ./streaming_demo
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock    = std::chrono::steady_clock;
using TimePoint= std::chrono::time_point<Clock>;
using Ms       = std::chrono::duration<double, std::milli>;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1 — UTF-8 boundary check
// ─────────────────────────────────────────────────────────────────────────────

/*
 * utf8_sequence_length
 *
 * Given the leading byte of a UTF-8 sequence, return the total number of
 * bytes in that sequence (1–4), or -1 for continuation bytes (invalid leaders).
 *
 * UTF-8 byte structure:
 *   0xxxxxxx             → 1 byte  (ASCII, U+0000–U+007F)
 *   110xxxxx 10xxxxxx    → 2 bytes (U+0080–U+07FF)
 *   1110xxxx 10xxxxxx×2  → 3 bytes (U+0800–U+FFFF)
 *   11110xxx 10xxxxxx×3  → 4 bytes (U+10000–U+10FFFF)
 *   10xxxxxx             → continuation byte (not a valid leader)
 */
int utf8_sequence_length(uint8_t lead_byte) {
    if ((lead_byte & 0x80) == 0x00) return 1;   // 0xxxxxxx — ASCII
    if ((lead_byte & 0xE0) == 0xC0) return 2;   // 110xxxxx
    if ((lead_byte & 0xF0) == 0xE0) return 3;   // 1110xxxx
    if ((lead_byte & 0xF8) == 0xF0) return 4;   // 11110xxx
    return -1;                                    // continuation byte
}

bool is_continuation_byte(uint8_t b) {
    return (b & 0xC0) == 0x80;   // 10xxxxxx
}

void demo_utf8_boundary_check() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 1: UTF-8 Boundary Check\n";
    std::cout << std::string(65, '=') << "\n\n";

    struct Case {
        std::string label;
        uint8_t     lead_byte;
        int         expected_len;
    };

    std::vector<Case> cases = {
        {"ASCII 'A' (0x41)",       0x41, 1},
        {"ASCII space (0x20)",     0x20, 1},
        {"2-byte lead 'é' (0xC3)", 0xC3, 2},
        {"3-byte lead '中' (0xE4)", 0xE4, 3},
        {"4-byte lead '😀'(0xF0)", 0xF0, 4},
        {"Continuation (0x80)",    0x80, -1},
        {"Continuation (0xA9)",    0xA9, -1},
    };

    std::cout << "  " << std::left << std::setw(28) << "Case"
              << std::right << std::setw(8) << "Byte"
              << std::setw(10) << "Seq Len"
              << "  Action\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    for (auto& c : cases) {
        int len = utf8_sequence_length(c.lead_byte);
        std::string action = (len == 1)  ? "Emit immediately" :
                             (len == 2)  ? "Buffer, wait for 1 more byte" :
                             (len == 3)  ? "Buffer, wait for 2 more bytes":
                             (len == 4)  ? "Buffer, wait for 3 more bytes":
                                           "Skip (invalid leader)";
        std::cout << "  " << std::left << std::setw(28) << c.label
                  << "  0x" << std::hex << std::uppercase << std::setw(2)
                  << std::setfill('0') << (int)c.lead_byte
                  << std::dec << std::setfill(' ')
                  << std::setw(8) << len
                  << "  " << action << "\n";
        assert(len == c.expected_len);
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2 — Streaming detokenizer
// ─────────────────────────────────────────────────────────────────────────────

class StreamingDetokenizer {
public:
    explicit StreamingDetokenizer(bool suppress_special = true)
        : suppress_special_(suppress_special), buf_len_(0), expected_len_(0) {}

    /*
     * feed: ingest raw bytes from one token.
     * Returns complete UTF-8 characters ready to emit.
     * May return empty string if bytes form an incomplete sequence.
     */
    std::string feed(const uint8_t* bytes, int n_bytes) {
        std::string emitted;
        for (int i = 0; i < n_bytes; ++i) {
            uint8_t b = bytes[i];
            if (buf_len_ == 0) {
                // Leading byte of a new character
                int seq_len = utf8_sequence_length(b);
                if (seq_len < 0) continue;  // invalid — skip
                expected_len_ = seq_len;
                buf_[buf_len_++] = b;
            } else {
                // Continuation byte
                if (!is_continuation_byte(b)) {
                    // Encoding error — reset and start over
                    buf_len_ = 0;
                    i--;  // re-process this byte as a leader
                    continue;
                }
                buf_[buf_len_++] = b;
            }

            if (buf_len_ == expected_len_) {
                // Complete character — emit it
                std::string ch(reinterpret_cast<char*>(buf_), buf_len_);
                if (!suppress_special_ || !is_special_token(ch)) {
                    emitted += ch;
                }
                buf_len_ = 0;
                expected_len_ = 0;
            }
        }
        return emitted;
    }

    std::string feed(const std::string& s) {
        return feed(reinterpret_cast<const uint8_t*>(s.data()),
                    static_cast<int>(s.size()));
    }

    std::string flush() {
        if (buf_len_ > 0) {
            // Force-emit whatever is buffered (replacement char for incomplete)
            std::string result(reinterpret_cast<char*>(buf_), buf_len_);
            buf_len_ = 0;
            return result;
        }
        return "";
    }

private:
    bool    suppress_special_;
    uint8_t buf_[4];
    int     buf_len_;
    int     expected_len_;

    static bool is_special_token(const std::string& s) {
        static const std::vector<std::string> specials = {
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
        };
        for (auto& sp : specials) if (s == sp) return true;
        return false;
    }
};

void demo_streaming_detokenizer() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 2: Streaming Detokenizer\n";
    std::cout << std::string(65, '=') << "\n\n";

    struct Case {
        std::string label;
        std::vector<std::string> token_bytes;  // raw bytes per token
    };

    std::vector<Case> cases = {
        {"ASCII safe",        {"The ", "cat ", "sat."}},
        {"2-byte char split", {"caf\xC3", "\xA9"}},        // "café" across tokens
        {"3-byte char split", {"\xE4\xB8", "\xAD"}},       // "中" split across tokens
        {"Special filtered",  {"hello", "<|eot_id|>", "!"}},
    };

    for (auto& c : cases) {
        StreamingDetokenizer det;
        std::cout << "  " << c.label << "\n";
        std::string full_output;
        for (size_t i = 0; i < c.token_bytes.size(); ++i) {
            std::string emitted = det.feed(c.token_bytes[i]);
            std::cout << "    Token " << i << ": " << std::setw(15) << std::left
                      << ("\"" + c.token_bytes[i] + "\"")
                      << " → emitted: "
                      << (emitted.empty() ? "(buffered)" : "\"" + emitted + "\"") << "\n";
            full_output += emitted;
        }
        std::string flushed = det.flush();
        if (!flushed.empty()) {
            std::cout << "    [flush] → \"" << flushed << "\"\n";
            full_output += flushed;
        }
        std::cout << "    Full output: \"" << full_output << "\"\n\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3 — SSE frame builder
// ─────────────────────────────────────────────────────────────────────────────

/*
 * Minimal JSON string escaper (for SSE payload construction).
 * A production implementation would use a proper JSON library.
 */
std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

std::string build_sse_frame(const std::string& content,
                             const std::string& request_id,
                             bool is_final = false) {
    // OpenAI-compatible chunk format
    std::ostringstream ss;
    ss << "data: {\"id\":\"" << request_id << "\","
       << "\"object\":\"chat.completion.chunk\","
       << "\"choices\":[{\"index\":0,"
       << "\"delta\":{" << (content.empty() ? "" : "\"content\":\"" + json_escape(content) + "\"") << "},"
       << "\"finish_reason\":" << (is_final ? "\"stop\"" : "null")
       << "}]}\n\n";
    return ss.str();
}

void demo_sse_builder() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 3: SSE Frame Builder\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<std::string> tokens = {"The", " quick", " brown", " fox"};

    std::cout << "  Token stream → SSE frames:\n\n";
    for (size_t i = 0; i < tokens.size(); ++i) {
        bool final = (i == tokens.size() - 1);
        std::string frame = build_sse_frame(tokens[i], "cmpl-demo", final);
        std::cout << "  Token " << i << ": " << std::quoted(tokens[i]) << "\n";
        std::cout << "  Frame:  " << frame;
    }
    std::cout << "data: [DONE]\n\n";
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4 — Annotated llama.cpp streaming decode loop
// ─────────────────────────────────────────────────────────────────────────────

void demo_llamacpp_stream_loop() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 4: Annotated llama.cpp Streaming Loop\n";
    std::cout << std::string(65, '=') << "\n\n";

    const char* code = R"code(
  // ── llama.cpp streaming decode loop (real C API) ─────────────────────
  // (requires #include "llama.h" + linking against libllama)

  StreamingDetokenizer det(/*suppress_special=*/true);
  char token_buf[256];

  for (int step = 0; step < max_tokens; ++step) {

      // ── 1. Forward pass ─────────────────────────────────────────────
      // Runs the transformer for one decode step.
      // GPU time: ~18 ms for LLaMA 3 8B at batch=32 on H100.
      if (llama_decode(ctx, batch) != 0) break;

      // ── 2. Sample next token ─────────────────────────────────────────
      // Applies sampler chain (Chapter 12): temperature → top-k → top-p
      llama_token next_id = llama_sampler_sample(smpl, ctx, /*idx=*/-1);
      llama_sampler_accept(smpl, next_id);  // update penalty history

      // ── 3. EOS check ─────────────────────────────────────────────────
      // llama_token_is_eog returns true for EOS, EOT, and end-of-generation
      // special tokens. Always check BEFORE detokenizing.
      if (llama_token_is_eog(model, next_id)) break;

      // ── 4. Detokenize ────────────────────────────────────────────────
      // Converts token ID to raw UTF-8 bytes.
      // n_chars may be 0 for special tokens, or 1-4 for regular tokens.
      // special=false: suppress special token strings from output.
      int n_chars = llama_token_to_piece(
          model, next_id, token_buf, sizeof(token_buf), 0, /*special=*/false
      );

      if (n_chars > 0) {
          // ── 5. UTF-8 boundary-safe emit ──────────────────────────────
          // det.feed() accumulates bytes and returns only complete chars.
          std::string text = det.feed(
              reinterpret_cast<const uint8_t*>(token_buf), n_chars
          );

          if (!text.empty()) {
              // ── 6. Write to output ───────────────────────────────────
              // For terminal output: write + fflush immediately.
              // Without fflush, stdio buffers output in 4 KB chunks.
              std::cout << text << std::flush;

              // For HTTP streaming: build SSE frame and write to socket.
              // std::string frame = build_sse_frame(text, request_id);
              // send_to_client(socket_fd, frame);
          }
      }

      // ── 7. Prepare next batch ────────────────────────────────────────
      // Single-token batch for the next decode step.
      llama_batch_clear(batch);
      llama_batch_add(batch, next_id, n_prompt + step, {0}, /*logits=*/true);
  }

  // Flush any remaining buffered UTF-8 bytes at EOS
  std::string remaining = det.flush();
  if (!remaining.empty()) std::cout << remaining << std::flush;
  // ──────────────────────────────────────────────────────────────────────
)code";

    std::cout << code << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5 — Chunked HTTP response simulator
// ─────────────────────────────────────────────────────────────────────────────

void demo_chunked_http() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 5: Chunked HTTP Response Simulator\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<std::string> tokens = {"The", " cat", " sat"};
    std::string request_id = "cmpl-abc123";

    std::cout << "  HTTP/1.1 chunked transfer encoding (SSE over HTTP):\n\n";
    std::cout << "  HTTP/1.1 200 OK\n";
    std::cout << "  Content-Type: text/event-stream\n";
    std::cout << "  Transfer-Encoding: chunked\n";
    std::cout << "  Cache-Control: no-cache\n";
    std::cout << "  \n";

    for (size_t i = 0; i < tokens.size(); ++i) {
        bool final = (i == tokens.size() - 1);
        std::string frame = build_sse_frame(tokens[i], request_id, final);
        size_t chunk_size = frame.size();

        // Print chunk size in hex, then chunk data, then CRLF
        std::cout << "  " << std::hex << chunk_size << std::dec
                  << "\\r\\n\n";
        // Show the frame with \n\n visible
        std::string display = frame;
        for (auto& c : display) if (c == '\n') c = ' ';
        std::cout << "  " << display << "\\r\\n\n";
    }

    // Terminal chunk
    std::cout << "  0\\r\\n\n";
    std::cout << "  \\r\\n\n\n";

    std::cout << "  Note: chunk size is in hex (e.g., 0x1a = 26 bytes).\n";
    std::cout << "  The HTTP server library handles chunking automatically;\n";
    std::cout << "  application code just writes SSE frames to the socket.\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6 — End-to-end latency breakdown timer
// ─────────────────────────────────────────────────────────────────────────────

struct TokenTiming {
    std::string token;
    double gpu_ms;
    double sample_ms;
    double detok_ms;
    double send_ms;
    double total_ms() const { return gpu_ms + sample_ms + detok_ms + send_ms; }
};

std::vector<TokenTiming> simulate_token_timings(
    const std::vector<std::string>& tokens,
    double gpu_ms    = 18.0,
    double network_ms= 0.5)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(0.85, 1.15);

    std::vector<TokenTiming> result;
    for (auto& tok : tokens) {
        result.push_back({
            tok,
            gpu_ms    * jitter(rng),
            0.10      * jitter(rng),
            0.05      * jitter(rng),
            (0.30 + network_ms) * jitter(rng),
        });
    }
    return result;
}

void demo_latency_breakdown() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 6: End-to-End Latency Breakdown\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<std::string> tokens = {
        "The"," quick"," brown"," fox"," jumps"," over"," the"," lazy"," dog","."
    };

    for (auto& [label, net_ms] : std::vector<std::pair<std::string,double>>{
        {"LAN (0.5ms network)", 0.5},
        {"WAN (80ms network)",  80.0}
    }) {
        auto timings = simulate_token_timings(tokens, 18.0, net_ms);

        double sum_gpu=0, sum_sample=0, sum_detok=0, sum_send=0;
        for (auto& t : timings) {
            sum_gpu    += t.gpu_ms;
            sum_sample += t.sample_ms;
            sum_detok  += t.detok_ms;
            sum_send   += t.send_ms;
        }
        double total = sum_gpu + sum_sample + sum_detok + sum_send;
        int n = (int)timings.size();

        std::cout << "  " << label << "\n";
        std::cout << "  " << std::left << std::setw(10) << "Token"
                  << std::right << std::setw(10) << "GPU(ms)"
                  << std::setw(9) << "Samp"
                  << std::setw(8) << "Detok"
                  << std::setw(10) << "Send"
                  << std::setw(9) << "Total" << "\n";
        std::cout << "  " << std::string(58, '-') << "\n";

        for (auto& t : timings) {
            std::cout << "  " << std::left << std::setw(10) << t.token
                      << std::right << std::fixed << std::setprecision(1)
                      << std::setw(10) << t.gpu_ms
                      << std::setw(9)  << t.sample_ms
                      << std::setw(8)  << t.detok_ms
                      << std::setw(10) << t.send_ms
                      << std::setw(9)  << t.total_ms() << "\n";
        }
        std::cout << "\n";
        std::cout << "  Avg/token:  GPU=" << sum_gpu/n << "ms"
                  << "  Send=" << sum_send/n << "ms"
                  << "  Total=" << total/n << "ms\n";
        std::cout << "  GPU share: " << sum_gpu/total*100 << "%"
                  << "  Network share: " << sum_send/total*100 << "%\n";
        std::cout << "  Throughput: " << 1000.0/(total/n) << " tok/s\n\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7 — Backpressure ring buffer simulation
// ─────────────────────────────────────────────────────────────────────────────

void demo_backpressure() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 7: Backpressure Ring Buffer Simulation\n";
    std::cout << std::string(65, '=') << "\n\n";

    struct Config {
        std::string label;
        double producer_ms;  // time between produced tokens
        double consumer_ms;  // time between consumed tokens
    };

    std::vector<Config> configs = {
        {"Fast consumer (10ms)",  18.0, 10.0},
        {"Matched (18ms)",        18.0, 18.0},
        {"Slow consumer (50ms)",  18.0, 50.0},
        {"Very slow (200ms)",     18.0, 200.0},
    };

    int n_tokens = 20;

    std::cout << "  " << std::left << std::setw(28) << "Consumer speed"
              << std::right << std::setw(10) << "Max buf"
              << std::setw(10) << "Avg buf"
              << std::setw(14) << "Finish (ms)" << "\n";
    std::cout << "  " << std::string(62, '-') << "\n";

    for (auto& cfg : configs) {
        std::deque<int> buf;
        std::vector<int> buf_sizes;
        double t_produce = 0.0, t_consume = 0.0, t_now = 0.0;
        int produced = 0, consumed = 0;

        while (consumed < n_tokens) {
            while (produced < n_tokens && t_produce <= t_now) {
                buf.push_back(produced++);
                t_produce += cfg.producer_ms;
            }
            if (!buf.empty() && t_consume <= t_now) {
                buf.pop_front();
                consumed++;
                t_consume += cfg.consumer_ms;
            }
            buf_sizes.push_back((int)buf.size());
            t_now += 1.0;
        }

        int max_buf = *std::max_element(buf_sizes.begin(), buf_sizes.end());
        double avg_buf = std::accumulate(buf_sizes.begin(), buf_sizes.end(), 0.0)
                         / buf_sizes.size();

        std::cout << "  " << std::left << std::setw(28) << cfg.label
                  << std::right << std::setw(10) << max_buf
                  << std::fixed << std::setprecision(1)
                  << std::setw(10) << avg_buf
                  << std::setw(14) << t_now << "\n";
    }

    std::cout << "\n  Key insight: slow consumers increase buffer depth and total time.\n";
    std::cout << "  vLLM's engine.abort() drains the buffer instantly on disconnect.\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\nChapter 13 — Token Streaming: The Last Mile\n";
    std::cout << "Companion code: streaming_demo.cpp\n";
    std::cout << std::string(65, '=') << "\n\n";

    demo_utf8_boundary_check();
    demo_streaming_detokenizer();
    demo_sse_builder();
    demo_llamacpp_stream_loop();
    demo_chunked_http();
    demo_latency_breakdown();
    demo_backpressure();

    std::cout << std::string(65, '=') << "\n";
    std::cout << "Chapter 13 C++ demo complete.\n\n";
    std::cout << "Key results:\n";
    std::cout << "  * UTF-8 boundary: buffer until expected_len bytes accumulated\n";
    std::cout << "  * SSE frame: 'data: {...}\\n\\n' per token, '[DONE]' sentinel\n";
    std::cout << "  * llama.cpp: fflush after every token — no stdio buffering\n";
    std::cout << "  * LAN: GPU dominates; WAN: network dominates\n";
    std::cout << "  * Slow consumers: buffer grows; abort() frees KV blocks\n";
    std::cout << std::string(65, '=') << "\n";

    return 0;
}

```

