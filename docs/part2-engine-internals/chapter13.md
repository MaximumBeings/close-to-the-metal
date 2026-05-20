# Chapter 13 — Token Streaming: The Last Mile

> *"The model finishes computing in 20 milliseconds.
> The token appears on screen 33 milliseconds later.
> The difference is the last mile — detokenization, framing, and the wire.
> It is unglamorous work, and it determines whether the product feels fast."*

---

## 13.0 Why This Chapter Matters

Every previous chapter ended at the sampled token ID.  That integer — say,
`7492` — is not yet text.  It must be converted back to a string, wrapped in
an HTTP frame, pushed across a TCP connection, and rendered in a browser.
This final leg of the journey is **token streaming**.

At 30 tokens/s, each token arrives 33 ms apart.  If the streaming
infrastructure adds even 20 ms of buffering, users perceive the output as
stuttering rather than flowing.  At 50 000 concurrent users, a naïve
implementation that holds tokens in memory until the full response is ready
requires gigabytes of buffer — and makes every response feel like it takes
the full generation time to complete.

Streaming is the feature that makes LLM output feel alive.

**What you will understand after this chapter:**

- How detokenization converts token IDs back to text, and why multi-byte
  UTF-8 sequences require careful boundary handling.

- The end-to-end latency budget from GPU kernel to character on screen.
- How Server-Sent Events (SSE) work as a protocol and why they are preferred
  over WebSocket for LLM streaming.

- How vLLM's async generator streams `RequestOutput` objects, handles
  backpressure, and supports client cancellation.

- How llama.cpp's built-in HTTP server implements the same SSE protocol in
  C++, and how to build a streaming loop from the raw C API.

**What you need first:**

- Chapter 12 (sampling produces a token ID each step).
- Chapter 9 (the forward pass timing — the GPU side of the latency budget).

---

## 13.1 Detokenization  `[FOUNDATIONAL]`

Detokenization is the inverse of tokenization: it maps a sequence of integer
token IDs back to a UTF-8 string.

### 13.1.1 The vocabulary mapping

Every model ships with a vocabulary file (typically a JSON or BPE merge file)
that maps integer IDs to byte sequences.  For LLaMA 3 (128 256-token
vocabulary), the mapping is stored in `tokenizer.json` and loaded at startup.

```
Example LLaMA 3 token ID → bytes mapping (simplified):
  7492  → "the"      (3 ASCII bytes: 0x74 0x68 0x65)
  8    → " "         (1 byte: 0x20, space prefix)
  13   → "\n"        (1 byte: 0x0A, newline)
  42177 → "ně"       (3 bytes: 0x6E 0xC4 0x9B, "n" + Czech ě)
  128001 → "<|eot_id|>" (special token, maps to empty string in output)
```

Detokenization for a completed response is trivial: concatenate the byte
sequences for each token ID in order.  Streaming detokenization — producing
text character by character as tokens arrive — requires extra care.

### 13.1.2 The UTF-8 boundary problem

UTF-8 encodes non-ASCII characters using 2–4 bytes.  A token boundary can
fall in the **middle** of a multi-byte UTF-8 sequence.

```
UTF-8 encoding of "café":
  'c' → 0x63             (1 byte, ASCII)
  'a' → 0x61             (1 byte, ASCII)
  'f' → 0x66             (1 byte, ASCII)
  'é' → 0xC3 0xA9        (2 bytes, U+00E9)

If the tokenizer splits "café" as ["caf", "é"]:
  Token 1 bytes: 0x63 0x61 0x66          → "caf"  ← safe to emit
  Token 2 bytes: 0xC3 0xA9               → "é"    ← safe to emit (complete)

If the tokenizer splits "café" as ["cafe\xC3", "\xA9"]:
  Token 1 bytes: 0x63 0x61 0x66 0x65 0xC3  ← 0xC3 is the START of a 2-byte
                                              sequence — DO NOT emit yet
  Token 2 bytes: 0xA9                       ← completion byte — now emit "é"
```

The streaming detokenizer must buffer bytes until a complete UTF-8 character
is available before emitting to the client.

```
UTF-8 byte structure (reference):
────────────────────────────────────────────────────────────
1-byte:  0xxxxxxx              (U+0000 – U+007F, ASCII)
2-byte:  110xxxxx 10xxxxxx     (U+0080 – U+07FF)
3-byte:  1110xxxx 10xxxxxx 10xxxxxx  (U+0800 – U+FFFF)
4-byte:  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx  (U+10000 – U+10FFFF)

Leading byte determines total sequence length:
  0x00–0x7F  → 1 byte  (safe to emit immediately)
  0xC0–0xDF  → 2 bytes (buffer until 1 more continuation byte arrives)
  0xE0–0xEF  → 3 bytes (buffer until 2 more continuation bytes arrive)
  0xF0–0xF7  → 4 bytes (buffer until 3 more continuation bytes arrive)
  0x80–0xBF  → continuation byte (should not appear at sequence start)
────────────────────────────────────────────────────────────
```

### 13.1.3 Special tokens

Special tokens — `<|begin_of_text|>`, `<|eot_id|>`, `<|end_of_text|>` in
LLaMA 3 — map to empty strings in the output.  They must be recognized and
suppressed during detokenization, not passed to the client as literal bytes.

`[COMMON TRAP]` — If a detokenizer does not strip special tokens, the client
receives raw strings like `<|eot_id|>` appended to the end of the response.
This breaks JSON parsing, corrupts downstream structured output, and confuses
users.  Always maintain a set of special token IDs and emit nothing for them.

---

## 13.2 End-to-End Latency Budget  `[FOUNDATIONAL]`

Where does the time go between a token being sampled and a character appearing
on screen?

```
WORKED EXAMPLE 13.1 — End-to-End Token Latency Breakdown
─────────────────────────────────────────────────────────────────────
Setup:
  Model:    LLaMA 3 8B on H100 SXM5
  Batch:    32 concurrent users in decode
  Network:  LAN (1 Gbps, same data center)
  Client:   Browser in same region

Breakdown per token:

  ┌──────────────────────────────────────────────────────────┐
  │  Component               │  Time   │  Notes              │
  ├──────────────────────────┼─────────┼─────────────────────┤
  │  GPU forward pass        │  18 ms  │  memory-BW bound    │
  │  Sampling (CPU)          │   0.1 ms│  top-k/p, temp      │
  │  Detokenization (CPU)    │   0.05 ms│ vocab lookup        │
  │  SSE frame construction  │   0.05 ms│ JSON encode + wrap  │
  │  Kernel write + TCP send │   0.3 ms│ socket buffer flush  │
  │  Network transit (LAN)   │   0.5 ms│ 1 Gbps, same DC     │
  │  Browser render          │   1.0 ms│ DOM update, layout  │
  ├──────────────────────────┼─────────┼─────────────────────┤
  │  TOTAL                   │  ~20 ms │                     │
  └──────────────────────────┴─────────┴─────────────────────┘

  Inter-token latency (ITL): ~20 ms → ~50 tokens/s perceived

Over WAN (cross-region, 100 ms RTT):
  Network transit: +80 ms  (dominant term)
  Total ITL:       ~100 ms → ~10 tokens/s perceived
  → Streaming is essential to hide this; without it users wait
    for the full response (potentially 10–60 seconds).
─────────────────────────────────────────────────────────────────────
```

The GPU is dominant on LAN; the network is dominant on WAN.  Streaming
eliminates the "full response wait" on both — the user sees the first token
as soon as it is generated, not after all tokens are done.

```
WITHOUT STREAMING (buffered response):
─────────────────────────────────────────────────────────────────────
t=0      Request arrives, prefill begins
t=100ms  Prefill complete, decode begins
t=100ms → t=3100ms  300 decode steps (300 tokens at 10 tok/s WAN)
t=3100ms Response sent in one HTTP response body
t=3200ms User sees text appear all at once

User experience: 3.2-second blank wait, then everything at once.

WITH STREAMING (SSE):
─────────────────────────────────────────────────────────────────────
t=0      Request arrives, prefill begins
t=100ms  First token sampled → SSE frame sent
t=200ms  User sees first token (100ms network latency)
t=210ms  Second token arrives
...
t=3200ms Final token, response complete

User experience: first character appears at 200ms, then tokens
flow continuously. Perceived as "fast" despite same total time.
```

---

## 13.3 Server-Sent Events (SSE)  `[FOUNDATIONAL]`

### 13.3.1 What SSE is

Server-Sent Events is an HTTP/1.1 protocol for pushing a stream of text
events from server to client over a single persistent connection.  It is
defined in the HTML5 specification and natively supported in all major
browsers via the `EventSource` API.

```
SSE wire format — one event per token:

HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"id":"cmpl-1","choices":[{"delta":{"content":"The"},"index":0}]}\n\n
data: {"id":"cmpl-1","choices":[{"delta":{"content":" cat"},"index":0}]}\n\n
data: {"id":"cmpl-1","choices":[{"delta":{"content":" sat"},"index":0}]}\n\n
data: [DONE]\n\n

Rules:
  • Each event starts with "data: " and ends with two newlines (\n\n)
  • The client reassembles the stream by splitting on \n\n
  • [DONE] is a sentinel that signals the end of generation
  • Events may optionally include "id:" and "event:" fields
```

### 13.3.2 SSE vs. WebSocket

```
┌─────────────────────┬───────────────────────────┬──────────────────────────┐
│ Property            │ SSE                        │ WebSocket                │
├─────────────────────┼───────────────────────────┼──────────────────────────┤
│ Direction           │ Server → client only       │ Bidirectional            │
│ Protocol            │ HTTP/1.1 or HTTP/2         │ Separate WS protocol     │
│ Browser support     │ EventSource API (native)   │ WebSocket API (native)   │
│ Proxy/CDN compat.   │ Excellent (plain HTTP)     │ Variable (needs upgrade) │
│ Reconnection        │ Automatic (browser handles)│ Manual                   │
│ Multiplexing (H2)   │ Yes (multiple streams/conn)│ No (one conn per session)│
│ Overhead per event  │ ~20 bytes (data: prefix)   │ 2–14 bytes (frame header)│
│ Use case fit        │ LLM token streaming        │ Chat, gaming, collab edit│
└─────────────────────┴───────────────────────────┴──────────────────────────┘
```

For LLM token streaming, SSE is the clear choice:

- Unidirectional (server pushes tokens; client does not send mid-stream).
- Works through HTTP proxies, CDNs, and load balancers without special config.
- Automatic reconnection handles transient network failures.
- The OpenAI API uses SSE, so ecosystem tools (SDKs, proxies) expect it.

`[COMMON TRAP]` — SSE over HTTP/1.1 requires one TCP connection per stream.
At 50 000 concurrent users, this is 50 000 open connections.  Ensure your
load balancer and upstream server support this concurrency.  Nginx requires
`proxy_buffering off` to pass SSE frames through without buffering.

### 13.3.3 The OpenAI-compatible SSE format

vLLM exposes an OpenAI-compatible API.  The SSE format mirrors OpenAI's:

```
# Streaming chat completion event (one per token):
data: {
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1714000000,
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "choices": [{
    "index": 0,
    "delta": {"content": "The"},
    "finish_reason": null
  }]
}

# Final event (finish_reason set, content may be empty):
data: {
  "id": "chatcmpl-abc123",
  ...
  "choices": [{
    "index": 0,
    "delta": {},
    "finish_reason": "stop"
  }]
}

# Sentinel:
data: [DONE]
```

---

## 13.4 vLLM's Async Streaming Architecture  `[FOUNDATIONAL]`

### 13.4.1 The async generator

vLLM exposes streaming via an **async generator** — a Python coroutine that
yields one `RequestOutput` object per scheduler step.

```python
# vLLM streaming — minimal example
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def stream_response(prompt: str):
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model="meta-llama/Meta-Llama-3-8B-Instruct")
    )
    params = SamplingParams(temperature=0.7, max_tokens=200)

    # generate() returns an async generator
    async for output in engine.generate(prompt, params, request_id="req-1"):
        # output.outputs[0].text contains the full text so far (cumulative)
        # output.outputs[0].token_ids[-1] is the most recently sampled token
        token_text = output.outputs[0].text
        finished   = output.finished
        yield token_text
        if finished:
            break
```

`RequestOutput` contains:

- `outputs`: list of `CompletionOutput` objects (one per beam if `n > 1`)
- `outputs[0].text`: cumulative decoded text so far
- `outputs[0].token_ids`: list of all token IDs generated so far
- `outputs[0].finish_reason`: `None`, `"stop"`, or `"length"`
- `finished`: `True` when generation is complete

### 13.4.2 Delta vs. cumulative text

`[COMMON TRAP]` — `output.outputs[0].text` is **cumulative** — it grows by
one token's worth of text at each step.  To stream only the new token, compute
the delta:

```python
prev_text = ""
async for output in engine.generate(prompt, params, request_id="req-1"):
    current_text = output.outputs[0].text
    new_token_text = current_text[len(prev_text):]   # delta only
    prev_text = current_text
    if new_token_text:
        print(new_token_text, end="", flush=True)
```

vLLM's OpenAI-compatible server handles this internally and sends only the
delta in each SSE event (`"delta": {"content": " cat"}`).

### 13.4.3 Backpressure

If the client reads events slower than the model generates tokens, the event
buffer grows.  vLLM handles this through Python's `asyncio` backpressure:

```
vLLM async backpressure model:
────────────────────────────────────────────────────────────────────
  AsyncLLMEngine runs in its own asyncio event loop (background thread)
  Output queue per request: bounded by max_num_seqs × token_buffer_size

  If client reads slowly:
    await generator.__anext__()  ← blocks until client is ready
    vLLM scheduler continues generating tokens
    Tokens queue up in the per-request output buffer

  If buffer fills:
    vLLM slows down generation for this request at the scheduler level
    Other requests are unaffected

  In practice:
    Token size: ~20 bytes per SSE event
    At 50 tok/s, 1-second buffer = 50 tokens × 20 bytes = 1 KB per user
    50 000 users: 50 MB of output buffers — manageable
```

### 13.4.4 Client cancellation

When a client disconnects mid-stream, the connection drops.  vLLM detects
this via the `asyncio` write failure and cancels the generator:

```python
# FastAPI / Starlette streaming response with cancellation handling
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(request: Request):
    async def event_generator():
        try:
            async for output in engine.generate(
                prompt, params, request_id=str(uuid.uuid4())
            ):
                delta = compute_delta(output)
                yield f"data: {delta}\n\n"
                if output.finished:
                    yield "data: [DONE]\n\n"
                    break
        except asyncio.CancelledError:
            # Client disconnected — abort the request
            await engine.abort(request_id)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
```

When `engine.abort(request_id)` is called, vLLM's scheduler removes the
request from the running queue, freeing its KV blocks immediately.  This
is critical for throughput — abandoned requests should not occupy GPU memory.

### 13.4.5 Measuring TTFT and ITL in the streaming path

```python
import time

async def stream_with_timing(prompt: str, engine, params):
    t_request = time.perf_counter()
    first_token_time = None
    token_times = []

    async for output in engine.generate(prompt, params, request_id="timed-1"):
        t_now = time.perf_counter()
        if first_token_time is None and output.outputs[0].token_ids:
            first_token_time = t_now
            ttft_ms = (first_token_time - t_request) * 1000
            print(f"TTFT: {ttft_ms:.1f} ms")
        elif first_token_time is not None:
            token_times.append(t_now)

        if output.finished:
            break

    if len(token_times) >= 2:
        itls = [(token_times[i] - token_times[i-1]) * 1000
                for i in range(1, len(token_times))]
        print(f"ITL p50: {sorted(itls)[len(itls)//2]:.1f} ms")
        print(f"ITL p99: {sorted(itls)[int(len(itls)*0.99)]:.1f} ms")
```

---

## 13.5 The Token Flow: ASCII End-to-End Diagram  `[FOUNDATIONAL]`

```
TOKEN STREAMING — END-TO-END FLOW (vLLM)
────────────────────────────────────────────────────────────────────────────

Step N, GPU side:
  ┌──────────────────────┐
  │   H100 GPU           │
  │                      │
  │  Forward pass        │  18 ms
  │  Attention + FFN     │
  │  Logits [vocab_size] │
  │         ↓            │
  │  Sampler (CPU)       │  0.1 ms
  │  token_id = 7492     │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  vLLM Scheduler      │
  │                      │
  │  RequestOutput built │
  │  text = "The"        │  0.05 ms
  │  finished = False    │
  └──────────┬───────────┘
             │  async generator yield
             ▼
  ┌──────────────────────┐
  │  FastAPI Handler     │
  │                      │
  │  delta = " cat"      │
  │  SSE frame encoded:  │  0.05 ms
  │  "data: {...}\n\n"   │
  └──────────┬───────────┘
             │  socket write
             ▼
  ┌──────────────────────┐
  │  TCP / TLS stack     │
  │                      │
  │  Kernel buffer flush │  0.3 ms
  │  Packet sent         │
  └──────────┬───────────┘
             │  ~0.5 ms LAN / ~80 ms WAN
             ▼
  ┌──────────────────────┐
  │  Browser             │
  │                      │
  │  EventSource onmsg   │
  │  DOM update: " cat"  │  1 ms
  │  Character rendered  │
  └──────────────────────┘

Total (LAN):  ~20 ms per token   → ~50 tokens/s perceived
Total (WAN):  ~100 ms per token  → ~10 tokens/s perceived
```

---

## 13.6 llama.cpp Streaming  `[FOUNDATIONAL]`

### 13.6.1 The raw C API decode loop

When using llama.cpp's C API directly, streaming is a manual loop:
decode one token → detokenize → write to output.

```c
// llama.cpp streaming loop (simplified C pseudocode)
char token_buf[256];

for (int step = 0; step < max_tokens; ++step) {
    // 1. Forward pass: llama_decode fills token logits
    llama_decode(ctx, batch);

    // 2. Sample next token using sampler chain (Chapter 12)
    llama_token next_id = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_accept(smpl, next_id);

    // 3. Check for EOS
    if (llama_token_is_eog(model, next_id)) break;

    // 4. Detokenize: token ID → UTF-8 bytes
    int n_chars = llama_token_to_piece(
        model,
        next_id,
        token_buf,
        sizeof(token_buf),
        /*lstrip=*/0,
        /*special=*/false   // suppress special tokens
    );

    // 5. Write to stdout immediately (streaming to terminal)
    // Note: check UTF-8 boundary before writing (see §13.1.2)
    fwrite(token_buf, 1, n_chars, stdout);
    fflush(stdout);    // ← critical: force write without buffering

    // 6. Prepare next batch: single token decode step
    llama_batch_clear(batch);
    llama_batch_add(batch, next_id, n_prompt + step, {0}, true);
}
```

`fflush(stdout)` after each token is critical.  Without it, the C runtime
buffers output in 4 KB chunks, producing the "jumpy" streaming experience
where many tokens appear at once after a pause.

### 13.6.2 llama.cpp's built-in HTTP server (llama-server)

llama.cpp ships a built-in HTTP server (`examples/server/`) that implements
the OpenAI-compatible SSE protocol in C++.  The server handles:

- Request parsing (JSON body, parameters)
- Concurrent sessions (one `llama_context` per slot)
- SSE frame construction and HTTP chunked transfer encoding
- Abort handling when clients disconnect

Key endpoints:
```
POST /completion          ← llama.cpp native format
POST /v1/chat/completions ← OpenAI-compatible format

Both support:  stream=true   for SSE streaming
               stream=false  for buffered (wait-for-all) response
```

### 13.6.3 Chunked transfer encoding

SSE responses use HTTP **chunked transfer encoding** — the server sends the
response body in chunks without knowing the total `Content-Length` upfront.

```
HTTP/1.1 200 OK
Transfer-Encoding: chunked
Content-Type: text/event-stream

1a\r\n                           ← chunk size in hex (26 bytes)
data: {"content":"The"}\n\n\r\n  ← chunk data
18\r\n                           ← next chunk size
data: {"content":" cat"}\n\n\r\n
0\r\n                            ← terminal chunk (zero length = done)
\r\n
```

This is handled transparently by the HTTP server library.  Application code
just writes SSE frames; chunked encoding is applied automatically.

### 13.6.4 UTF-8 handling in llama.cpp

`llama_token_to_piece` returns raw bytes, which may be an incomplete UTF-8
sequence.  The llama.cpp server accumulates bytes in a small ring buffer and
only emits them to the SSE frame once a complete UTF-8 character is assembled.

```c
// Simplified UTF-8 boundary check (llama.cpp server pattern)
static int utf8_continuation_bytes(unsigned char lead) {
    if ((lead & 0x80) == 0x00) return 0;   // ASCII — emit immediately
    if ((lead & 0xE0) == 0xC0) return 1;   // 2-byte sequence
    if ((lead & 0xF0) == 0xE0) return 2;   // 3-byte sequence
    if ((lead & 0xF8) == 0xF0) return 3;   // 4-byte sequence
    return -1;                              // continuation byte — should not lead
}

// In the streaming loop:
static char utf8_buf[4];
static int  utf8_buf_len = 0;

// After llama_token_to_piece fills token_buf:
for (int i = 0; i < n_chars; ++i) {
    utf8_buf[utf8_buf_len++] = token_buf[i];
    int need = utf8_continuation_bytes((unsigned char)utf8_buf[0]);
    if (utf8_buf_len == need + 1) {
        // Complete character — emit to SSE frame
        emit_sse_bytes(utf8_buf, utf8_buf_len);
        utf8_buf_len = 0;
    }
}
```

---

## 13.7 Consuming the vLLM Streaming API  `[FOUNDATIONAL]`

### 13.7.1 Python: direct HTTP SSE client

```python
import httpx
import json
import time

def stream_vllm(prompt: str, base_url: str = "http://localhost:8000") -> None:
    """
    Stream tokens from a running vLLM OpenAI-compatible server.
    Measures and prints TTFT and per-token timing.
    """
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 200,
    }

    t_start = time.perf_counter()
    first_token = True
    tokens_received = 0
    last_token_time = t_start

    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers={"Accept": "text/event-stream"},
        timeout=60.0,
    ) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]   # strip "data: " prefix
            if data_str == "[DONE]":
                break

            event = json.loads(data_str)
            delta = event["choices"][0]["delta"]
            content = delta.get("content", "")
            if not content:
                continue

            t_now = time.perf_counter()
            if first_token:
                ttft_ms = (t_now - t_start) * 1000
                print(f"\nTTFT: {ttft_ms:.1f} ms\n")
                first_token = False

            itl_ms = (t_now - last_token_time) * 1000
            last_token_time = t_now
            tokens_received += 1
            print(content, end="", flush=True)

    total_ms = (time.perf_counter() - t_start) * 1000
    print(f"\n\nTotal: {total_ms:.0f} ms | Tokens: {tokens_received} | "
          f"Avg ITL: {total_ms/max(tokens_received,1):.1f} ms/tok")
```

### 13.7.2 JavaScript: EventSource (browser-native)

```javascript
// Browser-native SSE — no library required
const eventSource = new EventSource('/v1/stream?prompt=Hello');

eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
        return;
    }
    const chunk = JSON.parse(event.data);
    const token = chunk.choices[0].delta.content ?? '';
    document.getElementById('output').textContent += token;
};

eventSource.onerror = () => {
    // EventSource auto-reconnects on error — close explicitly if done
    eventSource.close();
};
```

Note: `EventSource` does not support `POST` requests.  For POST-based
APIs (like OpenAI-compatible endpoints), use `fetch` with a readable stream:

```javascript
const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({messages: [...], stream: true}),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const text = decoder.decode(value, {stream: true});
    // Parse SSE lines from text
    for (const line of text.split('\n')) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
            const event = JSON.parse(line.slice(6));
            const token = event.choices[0].delta.content ?? '';
            appendToOutput(token);
        }
    }
}
```

---

## 13.8 Streaming at Scale — Operational Considerations  `[DEEP DIVE]`

### 13.8.1 Connection limits

Each streaming request holds one TCP connection open for the duration of
generation.  At 50 000 concurrent users:

```
WORKED EXAMPLE 13.2 — Connection and Memory Budget
─────────────────────────────────────────────────────────────────────
Given:
  Concurrent users:     50 000
  Avg generation time:  10 s (300 tokens at 30 tok/s)
  Avg response size:    300 tokens × 30 bytes/SSE = 9 KB
  OS per-connection overhead: ~4 KB (socket struct, buffers)

Memory for open connections:
  50 000 × 4 KB = 200 MB   (kernel socket buffers)

Output buffer (tokens not yet sent):
  Avg 2 tokens buffered × 30 bytes × 50 000 = 3 MB

Total connection overhead: ~203 MB — acceptable on a modern server

OS connection limits:
  Default Linux ulimit: 1 024 file descriptors per process
  Required:             50 000 + server fds ≈ 50 100
  Fix:  echo "* soft nofile 65536" >> /etc/security/limits.conf
        ulimit -n 65536
        (or set in systemd service: LimitNOFILE=65536)
─────────────────────────────────────────────────────────────────────
```

### 13.8.2 Nginx configuration for SSE passthrough

Without proper Nginx config, the proxy buffers the SSE response until the
connection closes — defeating the purpose of streaming entirely.

```nginx
# nginx.conf — SSE passthrough for vLLM
location /v1/ {
    proxy_pass         http://vllm_backend:8000;
    proxy_http_version 1.1;

    # Critical: disable proxy buffering for SSE
    proxy_buffering    off;
    proxy_cache        off;

    # Keep-alive for long-running streams
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;

    # SSE headers
    proxy_set_header   Connection "";
    add_header         X-Accel-Buffering no;

    # Pass client disconnect signal upstream
    proxy_ignore_client_abort off;
}
```

### 13.8.3 HTTP/2 multiplexing

HTTP/2 multiplexes multiple streams over a single TCP connection.  With SSE
over HTTP/2, 50 000 users can be served over far fewer TCP connections:

```
HTTP/1.1:  1 connection per SSE stream → 50 000 connections
HTTP/2:    ~100 streams per connection → ~500 connections

HTTP/2 benefits for SSE:
  ✓ Fewer OS file descriptors used
  ✓ Better TLS handshake amortization
  ✓ Header compression (HPACK) — small SSE events benefit

HTTP/2 requires:
  • TLS (HTTPS) — plaintext HTTP/2 (h2c) rarely supported by browsers
  • Server support (nginx, uvicorn, hypercorn with h2 extra)
  • Client support (all major browsers)
```

---

## 13.9 Comparison: vLLM vs. llama.cpp Streaming  `[FOUNDATIONAL]`

```
┌──────────────────────────┬─────────────────────────────┬──────────────────────────────┐
│ Feature                  │ vLLM                        │ llama.cpp                    │
├──────────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ Streaming mechanism      │ Async generator             │ Manual decode loop           │
│                          │ (AsyncLLMEngine.generate)   │ (llama_decode per token)     │
├──────────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ HTTP server              │ Built-in (FastAPI/uvicorn)  │ Built-in (llama-server)      │
│ SSE format               │ OpenAI-compatible           │ OpenAI-compatible            │
├──────────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ UTF-8 boundary handling  │ HuggingFace tokenizer       │ llama_token_to_piece +       │
│                          │ (handles internally)        │ manual buffer (§13.6.4)      │
├──────────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ Backpressure             │ asyncio + output queue      │ N/A (single user typical)    │
│ Client cancellation      │ engine.abort(request_id)    │ Connection close detection   │
├──────────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ Concurrent streams       │ Thousands (GPU batched)     │ Limited by context slots     │
│ Special token filtering  │ Automatic (tokenizer cfg)   │ special=false in token_piece │
├──────────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ TTFT visibility          │ Prometheus metric           │ llama_perf_context_print     │
│                          │ vllm:time_to_first_token    │ (manual timing wrapper)      │
└──────────────────────────┴─────────────────────────────┴──────────────────────────────┘
```

---

## 13.10 Code Listing  `[FOUNDATIONAL]`

See the [Chapter 13 companion code](../code/chapter_13.md) for:

- UTF-8 boundary detector and streaming detokenizer
- SSE frame parser (handles multi-line events, [DONE] sentinel)
- TTFT and ITL measurement harness
- Simulated vLLM streaming response with realistic timing
- Backpressure simulation: slow consumer vs. fast producer
- UTF-8 boundary check (utf8_continuation_bytes)
- Token-by-token streaming loop (annotated llama.cpp pattern)
- SSE frame builder: JSON encode + "data: ...\n\n" wrapper
- Chunked HTTP response simulator
- End-to-end latency breakdown timer

---

## 13.11 Summary

```
Key takeaways:

1. Detokenization maps token IDs to UTF-8 bytes via the vocabulary.
   Multi-byte characters can span token boundaries — buffer incomplete
   UTF-8 sequences until the leading byte's expected length is reached.

2. Special tokens (EOS, role markers) must be filtered out.
   Never pass raw special token strings to the client.

3. End-to-end per-token latency: ~20 ms LAN, ~100 ms WAN.
   The GPU forward pass dominates on LAN; network dominates on WAN.
   Streaming hides total generation time — users see the first token
   in milliseconds, not after the full response is ready.

4. SSE is the right protocol for LLM streaming:
   • Unidirectional, HTTP-native, works through proxies and CDNs.
   • Automatic browser reconnection on transient failures.
   • OpenAI-compatible format: "data: {...}\n\n", sentinel "[DONE]".

5. vLLM streams via an async generator (AsyncLLMEngine.generate).
   output.outputs[0].text is cumulative — compute delta for each event.
   engine.abort(request_id) on cancellation frees KV blocks immediately.

6. llama.cpp streams via a manual decode loop: llama_decode →
   llama_sampler_sample → llama_token_to_piece → fwrite + fflush.
   fflush after every token is essential; without it output is buffered.

7. At scale: set proxy_buffering off in Nginx, raise ulimit -n,
   and consider HTTP/2 to reduce connection count.
```

---

## 13.12 Self-Check Questions

1. A token's byte representation is `[0xC3, 0xA9]` (the character "é").
   The tokenizer splits it across two tokens: token A emits `[0xC3]` and
   token B emits `[0xA9]`.  What should the streaming detokenizer do
   after receiving token A's bytes?

2. You deploy vLLM behind Nginx and users report that tokens appear in
   bursts of 10–20 at once rather than one at a time.  What is the most
   likely cause and fix?

3. A client disconnects after receiving 50 tokens of a 500-token response.
   What should the vLLM server do, and what happens to the KV cache blocks
   for that request?

4. Your p99 TTFT is 2 500 ms but p50 is 120 ms.  Your p99 ITL is normal
   (25 ms).  What is the likely root cause?  (Hint: see Chapter 11.)

5. Explain why `fflush(stdout)` is required in the llama.cpp streaming loop
   but `print(..., flush=True)` is optional in a Python terminal application
   writing to stdout.

---

## Where We Go Next

Chapter 13 closes **Part II — Engine Internals, Side by Side**.  We have
now traced the complete lifetime of a request: admission by the scheduler
(Ch 7) → startup (Ch 8) → forward pass (Ch 9) → quantised weights (Ch 10)
→ prefill (Ch 11) → sampling (Ch 12) → streaming to the client (Ch 13).

**Part III — Production Configuration** begins with Chapter 14: the eight
vLLM parameters that control everything, with their llama.cpp equivalents,
and the interaction matrix that determines how changing one affects
throughput, latency, and memory.

*Next: Chapter 14 — The Eight vLLM Knobs + llama.cpp Equivalents*


---

## Chapter Summary

- **Server-Sent Events (SSE)**: vLLM's `/v1/chat/completions?stream=true` returns a chunked HTTP response with `data:` lines; each line contains a JSON delta with the newly generated token(s).
- **AsyncEngine architecture**: vLLM's streaming path uses an `AsyncLLMEngine` with an `asyncio` event loop; each request is an `AsyncGenerator` that yields `RequestOutput` objects.
- **Token streaming latency**: the first token is gated by TTFT (prefill); subsequent tokens arrive at ITL intervals (typically 20–80 ms depending on load and hardware).
- **Delta encoding**: streaming responses send only the new token text in each chunk, not the full response, minimizing network transfer and client-side buffer management.
- **Detokenization**: vLLM batches token IDs and detokenises incrementally; some tokens cannot be decoded until the next token arrives (e.g., multi-byte UTF-8 sequences).
- **llama.cpp streaming**: the `llama_token_to_str` function emits partial UTF-8 sequences; the `llama.cpp` server uses chunked Transfer-Encoding; the Python `llama-cpp-python` library wraps this with a `Generator`.
- **nginx buffering**: `proxy_buffering off` is required in the nginx config for streaming responses to reach the client without buffering delay.

---

## Self-Check Questions

1. A streaming response has TTFT = 800 ms and ITL = 35 ms. The response is 120 tokens. (a) When does the user see the first character? (b) How long until the full response is complete? *(Section 13.1)*

2. The vLLM `AsyncLLMEngine` yields a `RequestOutput` after every decode step. A client receives a `data:` SSE chunk with `delta.content = "the"`. What JSON structure does this chunk have? Draw the full SSE chunk including the `data:` prefix and the double-newline terminator. *(Section 13.2)*

3. A UTF-8 character requires 3 bytes. The model emits the three constituent token IDs in steps T, T+1, T+2. What does each streaming chunk contain, and how does vLLM handle the partial character at steps T and T+1? *(Section 13.4)*

4. You set `proxy_buffering on` in nginx in front of vLLM. Describe the user experience change and explain the buffering mechanism that causes it. *(Section 13.5)*

5. llama.cpp's `/v1/chat/completions` server uses chunked Transfer-Encoding rather than SSE. Name one client-side difference in how the response is parsed compared to an SSE stream. *(Section 13.3)*


---

## Worked Solutions

---

### Solution 1 — First character and full response timing

**Given:** TTFT=800 ms, ITL=35 ms/token, 120 tokens total

**Part (a) — When does the user see the first character?**

TTFT (Time to First Token) is exactly the time to the first visible character:

$$\textbf{800 ms} \text{ after the request is sent}$$

This includes: network round-trip, tokenization, prefill forward pass, and the decode of the first token. The user's screen shows nothing for 800 ms, then the first word appears.

**Part (b) — Time until the full response is complete.**

$$\text{total time} = \text{TTFT} + (\text{tokens} - 1) \times \text{ITL}$$
$$= 800 \text{ ms} + 119 \times 35 \text{ ms}$$
$$= 800 + 4{,}165 = \textbf{4,965 ms} \approx 5.0 \text{ seconds}$$

We subtract 1 from 120 because TTFT already accounts for generating the first token; the remaining 119 tokens are produced at ITL pace.

**Practical note:** 35 ms ITL corresponds to ~28.6 tokens/second — typical for a 7B model on an A10G GPU at batch size 16. The 800 ms TTFT suggests a 400–600 token prompt (prefill dominates TTFT for longer prompts).

---

### Solution 2 — SSE JSON chunk structure for delta.content = "the"

**What we need:** Full SSE wire format including JSON structure and terminators.

**The complete SSE chunk:**

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1699000000,"model":"meta-llama/Meta-Llama-3-8B-Instruct","choices":[{"index":0,"delta":{"content":"the"},"logprobs":null,"finish_reason":null}]}

```

**Breaking it down:**

| Component | Value | Meaning |
|-----------|-------|---------|
| `data: ` | literal prefix | SSE event data field |
| `"id"` | `"chatcmpl-abc123"` | Unique request ID |
| `"object"` | `"chat.completion.chunk"` | Streaming response type |
| `"choices[0].delta.content"` | `"the"` | The generated token text |
| `"finish_reason"` | `null` | Not null on last chunk (set to `"stop"` or `"length"`) |
| Trailing `\n\n` | (two newlines) | **SSE chunk terminator** — required |

**Why two newlines?**

The Server-Sent Events specification (RFC) uses a blank line (two consecutive `\n`) to signal the end of one event. Without it, the browser's EventSource API would not fire the `message` event. A common bug is sending only one newline, causing the client to buffer indefinitely waiting for the second.

---

### Solution 3 — UTF-8 partial character streaming

**Setup:** A 3-byte UTF-8 character (e.g., the Chinese character 你 = 0xE4 0xBD 0xA0) is emitted as three separate token steps.

**Step T — First byte emitted.**

The tokenizer can produce individual bytes as tokens for characters not in the vocabulary. At step T, the byte `0xE4` is emitted.

vLLM's text streamer calls the tokenizer's `decode([0xE4])` method — this returns an empty string or a replacement character (the bytes are not a complete codepoint).

**What the streamer does:** Buffer the incomplete bytes. **Send an empty delta** (`delta.content = ""`). The client sees no new character but does not stall.

**Step T+1 — Second byte emitted (0xBD).**

Buffer now contains [0xE4, 0xBD]. Still incomplete (requires 3 bytes for this codepoint). Streamer sends **empty delta** again.

**Step T+2 — Third byte emitted (0xA0).**

Buffer: [0xE4, 0xBD, 0xA0]. `decode([0xE4, 0xBD, 0xA0])` returns `"你"` — a complete, valid UTF-8 codepoint.

Streamer sends: `delta.content = "你"`.

**User experience:** No character appears for 2 steps (70 ms at 35 ms/ITL), then the Chinese character appears in one step. This 2-step delay is imperceptible to humans but visible in log analysis.

**Why this matters:** Sending raw bytes to the client without this buffering would produce garbled output (invalid UTF-8 sequences). The streamer's buffering is required for multilingual correctness.

---

### Solution 4 — proxy_buffering on: user experience and mechanism

**User experience change:**

Without `proxy_buffering off` (i.e., buffering is ON): the user sees **no streaming output**. Instead of tokens appearing one by one as the model generates them, the user sees a spinning cursor for the full response duration (~5 seconds in Solution 1), then the *complete* 120-token response appears all at once.

**Mechanism:**

1. vLLM sends SSE chunks (each one token) to nginx as they are produced.
2. nginx's proxy buffer collects these chunks in its internal buffer (default: 8 KB or 128 KB).
3. nginx waits until the buffer is full OR the upstream connection closes before forwarding to the client.
4. Since the full response is 120 tokens × ~10 bytes/token = ~1,200 bytes < 8 KB, nginx buffers the *entire* response before sending.
5. The client receives all 1,200 bytes in one TCP segment at t=5 seconds.

**Fix:**

```nginx
location /v1/ {
    proxy_pass http://vllm_backend;
    proxy_buffering off;           # forward chunks immediately
    proxy_cache off;
    proxy_set_header X-Accel-Buffering no;
}
```

Also set `X-Accel-Buffering: no` in the upstream response headers for belt-and-suspenders.

---

### Solution 5 — Chunked Transfer-Encoding vs SSE: client-side difference

**SSE (Server-Sent Events):**

The response uses `Content-Type: text/event-stream`. The client uses the browser's built-in `EventSource` API or a polyfill:

```javascript
const source = new EventSource('/v1/chat/completions');
source.onmessage = (event) => {
    const chunk = JSON.parse(event.data);
    console.log(chunk.choices[0].delta.content);
};
```

The `EventSource` API handles all parsing: it reads `data: {...}\n\n` lines, extracts the JSON, and fires `onmessage` for each complete event.

**Chunked Transfer-Encoding:**

The response uses `Transfer-Encoding: chunked`. Each chunk is prefixed with its length in hexadecimal:

```
5e

{"choices":[{"delta":{"content":"the"}}]}

0



```

The client must:

1. Read hex chunk length (`5e` = 94 bytes)
2. Read exactly 94 bytes of chunk body
3. Read the trailing `\r\n`
4. Repeat until chunk length = `0` (stream end marker)
5. Parse each chunk body as JSON

```javascript
// Simplified chunked stream reader
const reader = response.body.getReader();
let buffer = '';
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += new TextDecoder().decode(value);
    // Parse hex length, extract body, parse JSON...
}
```

**Key difference:** SSE provides a structured event protocol with built-in parsing (newline delimiters, `data:` prefix). Chunked transfer is raw byte streaming — the client must implement its own framing parser. SSE is simpler for browser clients; chunked is more flexible for server-to-server streaming with non-standard formats.

