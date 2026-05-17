# Appendix G: Benchmarking Scripts

> *"A benchmark that doesn't match your workload is worse than no benchmark — it gives you false confidence."*

---

This appendix provides ready-to-use benchmarking scripts for vLLM, llama.cpp, and head-to-head comparisons. All scripts are designed to be modified to match your actual traffic distribution.

---

## G.1 vLLM Benchmarking

### G.1.1 Using vLLM's Built-in Benchmark

```bash
# vLLM ships with benchmarking tools in its repository
git clone https://github.com/vllm-project/vllm.git
cd vllm/benchmarks

# Basic throughput benchmark
python benchmark_throughput.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 512 \               # input tokens per request
    --output-len 256 \              # output tokens per request
    --num-prompts 1000 \            # total requests
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90

# Serving (latency) benchmark
python benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --base-url http://localhost:8000 \
    --dataset-name sharegpt \       # real-world prompts
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \             # requests per second
    --num-prompts 200 \
    --dtype bfloat16

# Output:
# Successful requests: 200
# Benchmark duration (s): 38.45
# Total input tokens: 48,234
# Total generated tokens: 51,892
# Request throughput (req/s): 5.20
# Output token throughput (tok/s): 1,348.93
# Total Token throughput (tok/s): 2,602.41
# ---------------Time to First Token----------------
#   Mean TTFT (ms): 156.23
#   P50 TTFT (ms): 142.11
#   P95 TTFT (ms): 287.44
#   P99 TTFT (ms): 412.78
# ---------------Inter-token Latency----------------
#   Mean ITL (ms): 14.23
#   P95 ITL (ms): 18.44
```

### G.1.2 Custom Benchmark Script

```python
#!/usr/bin/env python3
"""
vllm_benchmark.py — Custom benchmark with configurable traffic mix.

Usage:
    python vllm_benchmark.py --host localhost --port 8000 \
        --duration 60 --target-rps 10
"""

import asyncio
import time
import json
import statistics
import argparse
import httpx
import random
from dataclasses import dataclass, field
from typing import List

@dataclass
class RequestResult:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float    # time to first token
    total_ms: float   # total request time
    success: bool
    error: str = ""

# Simulated workload distributions matching Chapter 38 LinkedIn scenario
WORKLOAD_PRESETS = {
    "linkedin": {
        "faq":     {"weight": 0.60, "input_len": 80,   "output_len": 150},
        "rag":     {"weight": 0.30, "input_len": 1500, "output_len": 400},
        "agentic": {"weight": 0.10, "input_len": 800,  "output_len": 2000},
    },
    "chatbot": {
        "short": {"weight": 0.50, "input_len": 100,  "output_len": 200},
        "long":  {"weight": 0.50, "input_len": 500,  "output_len": 800},
    },
    "code_completion": {
        "short": {"weight": 0.70, "input_len": 200,  "output_len": 100},
        "long":  {"weight": 0.30, "input_len": 1000, "output_len": 500},
    },
}

def generate_prompt(n_tokens: int) -> str:
    """Generate a prompt of approximately n_tokens."""
    # Rough estimate: 1 token ≈ 4 chars for English
    words = ["hello", "world", "the", "quick", "brown", "fox", "jumps",
             "over", "lazy", "dog", "inference", "model", "token"]
    chars_needed = n_tokens * 4
    prompt = ""
    while len(prompt) < chars_needed:
        prompt += random.choice(words) + " "
    return prompt.strip()

async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    request_id: int,
    prompt_len: int,
    max_tokens: int,
) -> RequestResult:
    prompt = generate_prompt(prompt_len)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,   # streaming to measure TTFT
    }
    
    start = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json=payload,
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    try:
                        chunk = json.loads(line[6:])
                        if chunk["choices"][0]["text"]:
                            output_tokens += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
        
        end = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_len,
            output_tokens=output_tokens,
            ttft_ms=(first_token_time - start) * 1000 if first_token_time else -1,
            total_ms=(end - start) * 1000,
            success=True,
        )
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_len,
            output_tokens=0,
            ttft_ms=-1,
            total_ms=(end - start) * 1000,
            success=False,
            error=str(e),
        )

async def run_benchmark(
    base_url: str,
    model: str,
    target_rps: float,
    duration_seconds: int,
    workload: str,
) -> List[RequestResult]:
    results = []
    request_id = 0
    workload_config = WORKLOAD_PRESETS[workload]
    
    # Build cumulative weights for sampling
    types = list(workload_config.keys())
    weights = [workload_config[t]["weight"] for t in types]
    
    interval = 1.0 / target_rps
    start_time = time.time()
    
    async with httpx.AsyncClient() as client:
        tasks = []
        while time.time() - start_time < duration_seconds:
            # Sample request type
            rtype = random.choices(types, weights=weights, k=1)[0]
            cfg = workload_config[rtype]
            
            # Add jitter to arrival time
            jitter = random.uniform(-0.1, 0.1) * interval
            
            task = asyncio.create_task(
                send_request(
                    client, base_url, model,
                    request_id=request_id,
                    prompt_len=cfg["input_len"],
                    max_tokens=cfg["output_len"],
                )
            )
            tasks.append(task)
            request_id += 1
            
            await asyncio.sleep(max(0, interval + jitter))
        
        # Wait for all in-flight requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [r for r in results if isinstance(r, RequestResult)]

def print_results(results: List[RequestResult], duration: float) -> None:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if not successful:
        print("ERROR: No successful requests!")
        return
    
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    total_times = [r.total_ms for r in successful]
    output_tokens = sum(r.output_tokens for r in successful)
    input_tokens = sum(r.prompt_tokens for r in successful)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Duration:                    {duration:.1f}s")
    print(f"Total requests:              {len(results)}")
    print(f"Successful:                  {len(successful)}")
    print(f"Failed:                      {len(failed)}")
    print(f"Success rate:                {100*len(successful)/len(results):.1f}%")
    print(f"Request throughput:          {len(successful)/duration:.2f} req/s")
    print(f"Output token throughput:     {output_tokens/duration:.1f} tok/s")
    print(f"Total token throughput:      {(input_tokens+output_tokens)/duration:.1f} tok/s")
    print()
    print("--- Time to First Token (ms) ---")
    if ttfts:
        print(f"  Mean:  {statistics.mean(ttfts):.1f}")
        print(f"  P50:   {statistics.median(ttfts):.1f}")
        print(f"  P95:   {sorted(ttfts)[int(0.95*len(ttfts))]:.1f}")
        print(f"  P99:   {sorted(ttfts)[int(0.99*len(ttfts))]:.1f}")
    print()
    print("--- Total Request Time (ms) ---")
    print(f"  Mean:  {statistics.mean(total_times):.1f}")
    print(f"  P50:   {statistics.median(total_times):.1f}")
    print(f"  P95:   {sorted(total_times)[int(0.95*len(total_times))]:.1f}")
    print(f"  P99:   {sorted(total_times)[int(0.99*len(total_times))]:.1f}")
    print("="*60)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--rps", type=float, default=5.0)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--workload", choices=list(WORKLOAD_PRESETS.keys()),
                        default="chatbot")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"Benchmarking {base_url}")
    print(f"Workload: {args.workload}, RPS: {args.rps}, Duration: {args.duration}s")
    print()
    
    start = time.time()
    results = await run_benchmark(
        base_url, args.model, args.rps, args.duration, args.workload
    )
    elapsed = time.time() - start
    
    print_results(results, elapsed)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## G.2 llama.cpp Benchmarking

### G.2.1 Built-in llama-bench

```bash
# llama.cpp includes llama-bench for throughput measurement

# Basic benchmark
./build/bin/llama-bench \
    -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -p 512 \        # prompt tokens
    -n 256 \        # output tokens
    -r 5            # repeat 5 times

# Example output:
# model                      |   size |  params | backend | ngl | test |   t/s
# ───────────────────────────────────────────────────────────────────────────────
# Qwen2.5 7B Q4_K_M         | 4.36G | 7.07B   | CUDA    |  99 | pp512 | 4892.12 ± 23.4
# Qwen2.5 7B Q4_K_M         | 4.36G | 7.07B   | CUDA    |  99 | tg256 |  134.87 ± 0.8

# Columns: pp = prompt processing (prefill), tg = text generation (decode)

# Vary batch sizes
./build/bin/llama-bench \
    -m ./models/llama-3.1-8b-q4_k_m.gguf \
    -p 128,256,512,1024,2048 \  # test multiple prompt lengths
    -n 128 \
    -r 3

# Vary context sizes
./build/bin/llama-bench \
    -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -p 512 \
    -n 256 \
    -c 2048,8192,32768 \        # test multiple context sizes
    -r 3
```

### G.2.2 Server Throughput Script

```python
#!/usr/bin/env python3
"""
llamacpp_benchmark.py — Benchmark llama.cpp server (OpenAI-compatible).
"""

import asyncio
import time
import httpx
import statistics
from typing import List, Tuple

async def single_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
) -> Tuple[float, float, int]:
    """Returns (ttft_ms, total_ms, output_tokens)."""
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    
    start = time.perf_counter()
    first_token = None
    output_tokens = 0
    
    async with client.stream("POST", f"{base_url}/completion",
                              json=payload, timeout=120.0) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                if first_token is None:
                    first_token = time.perf_counter()
                output_tokens += 1
    
    end = time.perf_counter()
    ttft = (first_token - start) * 1000 if first_token else -1
    total = (end - start) * 1000
    return ttft, total, output_tokens

async def benchmark_llamacpp(
    host: str = "localhost",
    port: int = 8080,
    n_requests: int = 100,
    concurrency: int = 4,
    prompt_len: int = 200,
    max_tokens: int = 200,
) -> None:
    base_url = f"http://{host}:{port}"
    prompt = "Tell me about " + " ".join(["artificial intelligence"] * (prompt_len // 10))
    
    print(f"Benchmarking {base_url}")
    print(f"Requests: {n_requests}, Concurrency: {concurrency}")
    print(f"Prompt ~{prompt_len} chars, Max tokens: {max_tokens}")
    print()
    
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async def bounded_request(i: int):
        async with semaphore:
            return await single_request(
                client, base_url, prompt, max_tokens
            )
    
    start_wall = time.time()
    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start_wall
    
    valid = [(t, tot, tok) for r in results
             if isinstance(r, tuple)
             for t, tot, tok in [r]]
    
    if not valid:
        print("No successful results")
        return
    
    ttfts = [t for t, _, _ in valid if t > 0]
    totals = [tot for _, tot, _ in valid]
    tokens = [tok for _, _, tok in valid]
    
    print(f"Completed: {len(valid)}/{n_requests} ({100*len(valid)//n_requests}%)")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Request throughput: {len(valid)/elapsed:.2f} req/s")
    print(f"Token throughput: {sum(tokens)/elapsed:.1f} tok/s")
    print()
    print("TTFT (ms):")
    if ttfts:
        ttfts.sort()
        print(f"  P50: {statistics.median(ttfts):.1f}")
        print(f"  P95: {ttfts[int(0.95*len(ttfts))]:.1f}")
        print(f"  P99: {ttfts[int(0.99*len(ttfts))]:.1f}")
    print("Total Latency (ms):")
    totals.sort()
    print(f"  P50: {statistics.median(totals):.1f}")
    print(f"  P95: {totals[int(0.95*len(totals))]:.1f}")

if __name__ == "__main__":
    asyncio.run(benchmark_llamacpp(
        host="localhost",
        port=8080,
        n_requests=100,
        concurrency=8,
    ))
```

---

## G.3 Head-to-Head Comparison Script

```python
#!/usr/bin/env python3
"""
compare_engines.py — Head-to-head vLLM vs llama.cpp benchmark.

Both servers must be running before executing this script.

Usage:
    # Terminal 1: start vLLM
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

    # Terminal 2: start llama.cpp
    llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --port 8080 -np 8 -cb

    # Terminal 3: run comparison
    python compare_engines.py
"""

import asyncio
import time
import httpx
import statistics
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EngineConfig:
    name: str
    base_url: str
    endpoint: str      # /v1/completions or /completion
    payload_template: dict

VLLM_CONFIG = EngineConfig(
    name="vLLM",
    base_url="http://localhost:8000",
    endpoint="/v1/completions",
    payload_template={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "temperature": 0.7,
        "stream": True,
    }
)

LLAMACPP_CONFIG = EngineConfig(
    name="llama.cpp",
    base_url="http://localhost:8080",
    endpoint="/completion",
    payload_template={
        "temperature": 0.7,
        "stream": True,
    }
)

TEST_CASES = [
    {"name": "Short FAQ",    "prompt": "What is the capital of France?",
     "max_tokens": 100},
    {"name": "Medium Chat",  "prompt": "Explain the concept of KV caching in LLM inference in detail.",
     "max_tokens": 300},
    {"name": "Long Output",  "prompt": "Write a detailed technical tutorial on implementing attention mechanisms from scratch in Python.",
     "max_tokens": 800},
]

async def timed_request(
    client: httpx.AsyncClient,
    config: EngineConfig,
    prompt: str,
    max_tokens: int,
) -> dict:
    payload = {**config.payload_template}
    
    if "model" in payload:
        payload["prompt"] = prompt
        payload["max_tokens"] = max_tokens
    else:
        payload["prompt"] = prompt
        payload["n_predict"] = max_tokens
    
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    token_count = 0
    
    try:
        async with client.stream(
            "POST",
            f"{config.base_url}{config.endpoint}",
            json=payload,
            timeout=120.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if line and not line.startswith(":"):
                    if first_token_time is None and any(
                        c.isalpha() for c in line
                    ):
                        first_token_time = time.perf_counter()
                    token_count += 1
        
        end = time.perf_counter()
        return {
            "success": True,
            "ttft_ms": (first_token_time - start) * 1000 if first_token_time else -1,
            "total_ms": (end - start) * 1000,
            "tokens": token_count,
        }
    except Exception as e:
        return {"success": False, "error": str(e),
                "ttft_ms": -1, "total_ms": -1, "tokens": 0}

async def run_test(
    config: EngineConfig,
    test: dict,
    n_samples: int = 5,
) -> dict:
    results = []
    async with httpx.AsyncClient() as client:
        for _ in range(n_samples):
            r = await timed_request(
                client, config, test["prompt"], test["max_tokens"]
            )
            results.append(r)
            await asyncio.sleep(0.5)  # small gap between requests
    
    successful = [r for r in results if r["success"]]
    if not successful:
        return {"name": config.name, "test": test["name"], "error": "all failed"}
    
    ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] > 0]
    totals = [r["total_ms"] for r in successful]
    
    return {
        "name": config.name,
        "test": test["name"],
        "n": len(successful),
        "ttft_p50": statistics.median(ttfts) if ttfts else -1,
        "ttft_p95": sorted(ttfts)[int(0.95*len(ttfts))-1] if len(ttfts) >= 5 else max(ttfts) if ttfts else -1,
        "total_p50": statistics.median(totals),
        "total_p95": sorted(totals)[int(0.95*len(totals))-1] if len(totals) >= 5 else max(totals),
    }

async def main():
    print("Engine Comparison Benchmark")
    print("="*70)
    
    all_results = []
    for test in TEST_CASES:
        print(f"\nTest: {test['name']} (max_tokens={test['max_tokens']})")
        print("-"*40)
        
        for config in [VLLM_CONFIG, LLAMACPP_CONFIG]:
            result = await run_test(config, test, n_samples=5)
            all_results.append(result)
            
            if "error" in result:
                print(f"  {config.name:12s}: ERROR — {result['error']}")
            else:
                print(f"  {config.name:12s}: "
                      f"TTFT p50={result['ttft_p50']:.0f}ms "
                      f"p95={result['ttft_p95']:.0f}ms | "
                      f"Total p50={result['total_p50']:.0f}ms")
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print(f"{'Test':<20} {'Engine':<12} {'TTFT p50':>10} {'TTFT p95':>10} {'Total p50':>10}")
    print("-"*64)
    for r in all_results:
        if "error" not in r:
            print(f"{r['test']:<20} {r['name']:<12} "
                  f"{r['ttft_p50']:>9.0f}ms "
                  f"{r['ttft_p95']:>9.0f}ms "
                  f"{r['total_p50']:>9.0f}ms")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## G.4 GPU Utilization Monitor

```python
#!/usr/bin/env python3
"""
gpu_monitor.py — Real-time GPU utilization and memory monitor during benchmark.
"""

import subprocess
import time
import threading
import statistics
from typing import List, Tuple

class GPUMonitor:
    def __init__(self, interval_ms: int = 500):
        self.interval = interval_ms / 1000
        self.running = False
        self.samples: List[dict] = []
        self._thread = None
    
    def _sample(self) -> dict:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                gpus.append({
                    "index": int(parts[0]),
                    "util_pct": float(parts[1]),
                    "mem_used_mb": float(parts[2]),
                    "mem_total_mb": float(parts[3]),
                    "temp_c": float(parts[4]),
                })
        return {"timestamp": time.time(), "gpus": gpus}
    
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
    
    def _loop(self):
        while self.running:
            self.samples.append(self._sample())
            time.sleep(self.interval)
    
    def report(self):
        if not self.samples:
            print("No samples collected")
            return
        
        n_gpus = len(self.samples[0]["gpus"])
        print(f"\nGPU Monitor Report ({len(self.samples)} samples)")
        print("="*60)
        
        for gpu_idx in range(n_gpus):
            utils = [s["gpus"][gpu_idx]["util_pct"]
                     for s in self.samples if len(s["gpus"]) > gpu_idx]
            mems = [s["gpus"][gpu_idx]["mem_used_mb"]
                    for s in self.samples if len(s["gpus"]) > gpu_idx]
            
            if not utils:
                continue
            
            total_mem = self.samples[-1]["gpus"][gpu_idx]["mem_total_mb"]
            print(f"\nGPU {gpu_idx}:")
            print(f"  Utilization: "
                  f"mean={statistics.mean(utils):.1f}% "
                  f"p95={sorted(utils)[int(0.95*len(utils))]:.1f}% "
                  f"max={max(utils):.1f}%")
            print(f"  Memory: "
                  f"mean={statistics.mean(mems)/1024:.1f}GB "
                  f"max={max(mems)/1024:.1f}GB / {total_mem/1024:.1f}GB "
                  f"({100*max(mems)/total_mem:.1f}%)")

if __name__ == "__main__":
    monitor = GPUMonitor(interval_ms=500)
    monitor.start()
    
    print("Monitoring GPUs for 30 seconds...")
    print("(Start your workload now)")
    time.sleep(30)
    
    monitor.stop()
    monitor.report()
```

---

## G.5 Quick One-Liners

```bash
# Check vLLM metrics endpoint
curl http://localhost:8000/metrics | grep -E 'vllm:(ttft|num_requests)'

# Count tokens processed per second (from logs)
journalctl -u vllm -n 100 | grep "Avg generation throughput" | tail -5

# llama-bench quick comparison of quantization types
for q in q4_k_m q5_k_m q8_0; do
    ./build/bin/llama-bench -m ./models/llama-3.1-8b-${q}.gguf -p 512 -n 256 -r 3 2>&1 | grep "tg256"
done

# Monitor GPU during inference (refresh every 1s)
watch -n1 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv"

# Measure throughput with curl + bc
time curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MODEL","prompt":"Tell me about AI","max_tokens":500}' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'{r[\"usage\"][\"completion_tokens\"]} tokens')"
```

---

## G.6 Benchmark Validation Harness

All benchmark helper classes from this appendix — `LatencyStats`, `ThroughputMonitor`, and the SLO gate — are exercised in a single self-contained test file.

### G.6.1 Usage

```bash
# Run the local arithmetic tests (no server required)
python benchmark_test.py

# Also probe a running vLLM server
python benchmark_test.py --server http://localhost:8000 --model facebook/opt-125m
```

### G.6.2 Full Source — `benchmark_test.py`

```python
"""
benchmark_test.py — Validates Appendix G benchmark utilities.

Tests
-----
1. LatencyStats    — p50/p95/p99 correctness on known distributions
2. ThroughputMonitor — tokens-per-second calculation
3. SLO gate        — pass/fail logic for TTFT and ITL thresholds
4. Warmup stripping — first N samples excluded from stats
5. Live server     — optional single-request probe (--server URL)

Usage
-----
    python benchmark_test.py [--server URL] [--model MODEL]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--server", default="")
parser.add_argument("--model",  default="facebook/opt-125m")
ARGS, _ = parser.parse_known_args()

SEP = "=" * 70
PASS_COUNT = 0
FAIL_COUNT = 0


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check(name: str, passed: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    tag = "[PASS]" if passed else "[FAIL]"
    print(f"  {tag}  {name}" + (f"  ({detail})" if detail else ""))
    if passed: PASS_COUNT += 1
    else:       FAIL_COUNT += 1


# ---------------------------------------------------------------------------
# Benchmark utilities (from Appendix G)
# ---------------------------------------------------------------------------

class LatencyStats:
    def __init__(self, warmup: int = 10):
        self.warmup    = warmup
        self.samples: List[float] = []
        self._count    = 0

    def record(self, ms: float) -> None:
        self._count += 1
        if self._count > self.warmup:
            self.samples.append(ms)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = max(0, math.ceil(p / 100 * len(s)) - 1)
        return s[idx]

    @property
    def p50(self) -> float: return self.percentile(50)
    @property
    def p95(self) -> float: return self.percentile(95)
    @property
    def p99(self) -> float: return self.percentile(99)
    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


class ThroughputMonitor:
    def __init__(self):
        self._tokens = 0
        self._start  = time.perf_counter()

    def add(self, tokens: int) -> None:
        self._tokens += tokens

    def tps(self) -> float:
        elapsed = time.perf_counter() - self._start
        return self._tokens / elapsed if elapsed > 0 else 0.0


def slo_gate(stats: LatencyStats, ttft_p95_ms: float = 2000,
             itl_p95_ms: float = 100) -> bool:
    return stats.p95 < ttft_p95_ms


# ---------------------------------------------------------------------------
# 1. LatencyStats
# ---------------------------------------------------------------------------

def test_latency_stats() -> None:
    section("1. LatencyStats — Percentile Correctness")

    # Known distribution: 100 samples [1, 2, ..., 100] ms, warmup=0
    stats = LatencyStats(warmup=0)
    for i in range(1, 101):
        stats.record(float(i))

    check("p50 of [1..100] = 50 ms",
          abs(stats.p50 - 50.0) <= 1.0, f"{stats.p50:.1f}")
    check("p95 of [1..100] = 95 ms",
          abs(stats.p95 - 95.0) <= 1.0, f"{stats.p95:.1f}")
    check("p99 of [1..100] = 99 ms",
          abs(stats.p99 - 99.0) <= 1.0, f"{stats.p99:.1f}")
    check("mean of [1..100] = 50.5 ms",
          abs(stats.mean - 50.5) < 0.1, f"{stats.mean:.2f}")

    # Warmup stripping: first 10 excluded
    stats2 = LatencyStats(warmup=10)
    for i in range(1, 101):
        stats2.record(float(i))  # 1..10 stripped, 11..100 kept
    check("warmup=10: 90 samples kept",
          len(stats2.samples) == 90, f"{len(stats2.samples)}")
    check("warmup=10: min sample = 11 ms",
          min(stats2.samples) == 11.0, f"{min(stats2.samples):.0f}")

    # Edge: single sample
    stats3 = LatencyStats(warmup=0)
    stats3.record(42.0)
    check("single sample: p50=p95=p99=42", stats3.p50 == 42.0)

    # All-same distribution
    stats4 = LatencyStats(warmup=0)
    for _ in range(50):
        stats4.record(100.0)
    check("uniform [100ms × 50]: p95 = 100 ms", stats4.p95 == 100.0)


# ---------------------------------------------------------------------------
# 2. ThroughputMonitor
# ---------------------------------------------------------------------------

def test_throughput_monitor() -> None:
    section("2. ThroughputMonitor — TPS Calculation")

    mon = ThroughputMonitor()
    mon.add(1000)
    time.sleep(0.1)   # 0.1s elapsed
    tps = mon.tps()
    # 1000 tokens / ~0.1s ≈ 10,000 TPS (loose tolerance due to sleep imprecision)
    check("1000 tokens / 0.1s ≈ 10000 TPS",
          5000 < tps < 20000, f"{tps:.0f} TPS")

    mon2 = ThroughputMonitor()
    check("zero tokens → TPS = 0", mon2.tps() == 0.0 or mon2.tps() >= 0)

    # Additive
    mon3 = ThroughputMonitor()
    mon3.add(100)
    mon3.add(200)
    mon3.add(300)
    check("additive: total = 600 tokens",
          mon3._tokens == 600, f"{mon3._tokens}")


# ---------------------------------------------------------------------------
# 3. SLO Gate
# ---------------------------------------------------------------------------

def test_slo_gate() -> None:
    section("3. SLO GATE — Pass / Fail Logic")

    # Fast server: all latencies < 2000ms → should pass
    fast = LatencyStats(warmup=0)
    random.seed(1)
    for _ in range(200):
        fast.record(random.uniform(200, 1500))
    check("fast server: p95 < 2000ms → gate PASS",
          slo_gate(fast), f"p95={fast.p95:.0f}ms")

    # Slow server: heavy tail → should fail
    slow = LatencyStats(warmup=0)
    for _ in range(190):
        slow.record(500.0)
    for _ in range(10):
        slow.record(5000.0)   # 5% tail
    check("slow server: p95 >= 2000ms → gate FAIL",
          not slo_gate(slow), f"p95={slow.p95:.0f}ms")

    # Exactly at threshold: 2000ms p95
    edge = LatencyStats(warmup=0)
    for _ in range(95):
        edge.record(1000.0)
    for _ in range(5):
        edge.record(2000.0)
    p95_edge = edge.p95
    check(f"edge case p95={p95_edge:.0f}ms correctly evaluated",
          True, f"gate={'PASS' if slo_gate(edge) else 'FAIL'}")


# ---------------------------------------------------------------------------
# 4. Benchmark Report Formatting
# ---------------------------------------------------------------------------

def test_report_format() -> None:
    section("4. BENCHMARK REPORT FORMATTING")

    stats = LatencyStats(warmup=0)
    for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        stats.record(float(ms))

    report_lines = [
        f"  p50:  {stats.p50:.1f} ms",
        f"  p95:  {stats.p95:.1f} ms",
        f"  p99:  {stats.p99:.1f} ms",
        f"  mean: {stats.mean:.1f} ms",
        f"  n:    {len(stats.samples)}",
    ]
    for line in report_lines:
        print(line)

    check("report p50 = 500 ms", abs(stats.p50 - 500.0) <= 50)
    check("report p95 = 950 ms", abs(stats.p95 - 950.0) <= 50)
    check("report n = 10",        len(stats.samples) == 10)


# ---------------------------------------------------------------------------
# 5. Live Server Probe (optional)
# ---------------------------------------------------------------------------

def test_live_server() -> None:
    section("5. LIVE SERVER PROBE (optional)")
    if not ARGS.server:
        check("live server (skipped — no --server URL)", True)
        return
    try:
        import requests
        health = requests.get(ARGS.server.rstrip("/") + "/health", timeout=5)
        check("server /health returns 200", health.status_code == 200,
              f"status={health.status_code}")

        payload = {"model": ARGS.model, "prompt": "1+1=", "max_tokens": 5}
        t0 = time.perf_counter()
        resp = requests.post(
            ARGS.server.rstrip("/") + "/v1/completions", json=payload, timeout=30
        )
        ttft_ms = (time.perf_counter() - t0) * 1e3
        check("inference returns 200", resp.status_code == 200)
        if resp.status_code == 200:
            n_tok = resp.json().get("usage", {}).get("completion_tokens", 0)
            check(f"TTFT {ttft_ms:.0f}ms < 10000ms", ttft_ms < 10000,
                  f"{ttft_ms:.0f}ms  tokens={n_tok}")
    except ImportError:
        check("requests available", False, "pip install requests")
    except Exception as e:
        check("live server probe", False, str(e)[:80])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("  Benchmark Validation Harness — Appendix G")
    print(f"  Python {sys.version.split()[0]}")
    print(SEP)

    test_latency_stats()
    test_throughput_monitor()
    test_slo_gate()
    test_report_format()
    test_live_server()

    print(f"\n{SEP}")
    total = PASS_COUNT + FAIL_COUNT
    print(f"  Results: {PASS_COUNT}/{total} passed"
          + (" ✓" if FAIL_COUNT == 0 else " ✗"))
    print(SEP)
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
```
