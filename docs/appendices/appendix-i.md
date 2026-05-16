# Appendix I: C++ Build Patterns for Inference Extensions

> *"C++ extensions for inference follow a small set of recurring patterns. Master the patterns, and new extensions become incremental work."*

---

This appendix provides complete, compilable C++ patterns for extending inference engines. All code targets C++17 with standard library dependencies only, unless noted.

---

## I.1 Build System Setup

### I.1.1 CMakeLists.txt Template

```cmake
# CMakeLists.txt — Template for inference extension
cmake_minimum_required(VERSION 3.18)
project(inference_extension VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")

# Find CUDA (optional)
find_package(CUDAToolkit QUIET)

# ---- Pure C++ library ----
add_library(inference_core STATIC
    src/kv_cache.cpp
    src/tokenizer.cpp
    src/scheduler.cpp
)
target_include_directories(inference_core PUBLIC include/)

# ---- CUDA extension (if available) ----
if(CUDAToolkit_FOUND)
    add_library(cuda_kernels STATIC
        kernels/attention.cu
        kernels/gemv.cu
    )
    set_target_properties(cuda_kernels PROPERTIES
        CUDA_ARCHITECTURES "80;86;90"  # A100, A40, H100
    )
    target_link_libraries(cuda_kernels CUDA::cudart CUDA::cublas)
    target_link_libraries(inference_core cuda_kernels)
    add_compile_definitions(inference_core HAS_CUDA=1)
endif()

# ---- Benchmark executable ----
add_executable(bench src/bench.cpp)
target_link_libraries(bench inference_core)

# ---- Test executable ----
add_executable(test_runner tests/test_main.cpp)
target_link_libraries(test_runner inference_core)
enable_testing()
add_test(NAME unit_tests COMMAND test_runner)
```

### I.1.2 Makefile Alternative

```makefile
# Makefile — Simple alternative for quick builds
CXX      = g++
NVCC     = nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -I include/
NVCCFLAGS = -std=c++17 -O2 -arch=sm_90 -I include/

SRC      = $(wildcard src/*.cpp)
CUDA_SRC = $(wildcard kernels/*.cu)
OBJ      = $(SRC:.cpp=.o) $(CUDA_SRC:.cu=.o)

.PHONY: all clean test

all: bench

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bench: $(OBJ) src/bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -lcuda -lcudart -lcublas

clean:
	rm -f $(OBJ) bench

test: bench
	./bench --test
```

---

## I.2 KV Cache Implementation Pattern

```cpp
// include/kv_cache.h
#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cassert>
#include <memory>

// Block-based KV cache mirroring vLLM's PagedAttention design.
// Each block holds BLOCK_SIZE tokens for all layers.

constexpr int BLOCK_SIZE = 16;          // tokens per block
constexpr int MAX_BLOCKS = 8192;        // total available blocks

struct KVBlock {
    int block_id;
    int ref_count = 0;           // for copy-on-write / prefix sharing
    bool is_dirty = false;
    
    // For a model with n_layers layers, n_kv_heads KV heads, head_dim:
    // Key:   [BLOCK_SIZE, n_layers, n_kv_heads, head_dim]
    // Value: [BLOCK_SIZE, n_layers, n_kv_heads, head_dim]
    // Stored flat in this buffer.
    std::vector<float> key_data;
    std::vector<float> val_data;
    
    KVBlock(int id, int n_layers, int n_kv_heads, int head_dim)
        : block_id(id) {
        size_t sz = BLOCK_SIZE * n_layers * n_kv_heads * head_dim;
        key_data.resize(sz, 0.0f);
        val_data.resize(sz, 0.0f);
    }
};

class KVBlockManager {
public:
    int n_layers, n_kv_heads, head_dim;
    
    KVBlockManager(int n_layers, int n_kv_heads, int head_dim)
        : n_layers(n_layers), n_kv_heads(n_kv_heads), head_dim(head_dim) {
        // Pre-allocate block pool
        for (int i = 0; i < MAX_BLOCKS; i++) {
            free_blocks_.push_back(i);
            block_pool_.emplace_back(i, n_layers, n_kv_heads, head_dim);
        }
    }
    
    // Allocate N blocks, return their IDs
    std::vector<int> allocate(int n) {
        assert(n <= (int)free_blocks_.size() && "KV cache OOM");
        std::vector<int> allocated;
        for (int i = 0; i < n; i++) {
            int bid = free_blocks_.back();
            free_blocks_.pop_back();
            block_pool_[bid].ref_count = 1;
            allocated.push_back(bid);
        }
        return allocated;
    }
    
    // Free a block
    void free(int block_id) {
        auto& blk = block_pool_[block_id];
        assert(blk.ref_count > 0);
        blk.ref_count--;
        if (blk.ref_count == 0) {
            free_blocks_.push_back(block_id);
        }
    }
    
    // Get reference to block
    KVBlock& get(int block_id) {
        return block_pool_[block_id];
    }
    
    int num_free() const { return (int)free_blocks_.size(); }
    int num_used() const { return MAX_BLOCKS - num_free(); }
    
    // Number of blocks needed for seq_len tokens
    static int blocks_needed(int seq_len) {
        return (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

private:
    std::vector<int> free_blocks_;
    std::vector<KVBlock> block_pool_;
};

// Per-sequence KV cache state
struct SequenceKVState {
    int seq_id;
    int n_tokens = 0;             // tokens currently cached
    std::vector<int> block_ids;   // logical → physical block mapping
    
    SequenceKVState(int seq_id) : seq_id(seq_id) {}
    
    // Get logical block index for token position pos
    int logical_block(int pos) const {
        return pos / BLOCK_SIZE;
    }
    
    // Token offset within its block
    int block_offset(int pos) const {
        return pos % BLOCK_SIZE;
    }
    
    // Current last block (the one being written to)
    int last_block_id() const {
        assert(!block_ids.empty());
        return block_ids.back();
    }
    
    // Is the last block full?
    bool last_block_full() const {
        return (n_tokens % BLOCK_SIZE) == 0;
    }
};
```

---

## I.3 Scheduler Pattern

```cpp
// include/scheduler.h
#pragma once
#include <queue>
#include <vector>
#include <memory>
#include <optional>
#include <functional>

enum class RequestStatus {
    WAITING,      // in queue, not yet started
    RUNNING,      // currently generating tokens
    SWAPPED,      // preempted, KV on CPU
    FINISHED,     // generation complete
    ABORTED,      // cancelled
};

struct Request {
    int id;
    std::vector<int> prompt_tokens;
    int max_new_tokens;
    int tokens_generated = 0;
    RequestStatus status = RequestStatus::WAITING;
    float arrival_time;
    int priority = 0;             // higher = more urgent
    
    std::vector<int> generated_tokens;
    std::optional<std::string> error;
    
    bool is_done() const {
        return tokens_generated >= max_new_tokens;
    }
    
    int total_tokens() const {
        return (int)prompt_tokens.size() + tokens_generated;
    }
};

struct ScheduleBatch {
    std::vector<int> prefill_request_ids;   // requests to run prefill
    std::vector<int> decode_request_ids;    // requests to run decode
    int total_tokens() const {
        // approximation
        return (int)(prefill_request_ids.size() + decode_request_ids.size());
    }
};

class ContinuousBatchScheduler {
public:
    int max_batch_tokens;
    int max_sequences;
    
    ContinuousBatchScheduler(int max_batch_tokens = 2048, int max_sequences = 64)
        : max_batch_tokens(max_batch_tokens), max_sequences(max_sequences) {}
    
    void add_request(std::shared_ptr<Request> req) {
        waiting_queue_.push(req);
    }
    
    // Schedule next batch: returns requests to run
    ScheduleBatch schedule() {
        ScheduleBatch batch;
        int token_budget = max_batch_tokens;
        
        // First: schedule decode for all running sequences
        for (auto& [id, req] : running_) {
            if (token_budget <= 0) break;
            if ((int)batch.decode_request_ids.size() >= max_sequences) break;
            batch.decode_request_ids.push_back(id);
            token_budget -= 1;  // decode takes 1 token per sequence
        }
        
        // Second: admit new requests for prefill if budget allows
        while (!waiting_queue_.empty() && token_budget > 0) {
            auto req = waiting_queue_.top();
            int prefill_tokens = (int)req->prompt_tokens.size();
            
            if (prefill_tokens > token_budget) {
                // Can't fit this request's prefill in remaining budget
                // With chunked prefill we could split — simplified here
                break;
            }
            
            waiting_queue_.pop();
            req->status = RequestStatus::RUNNING;
            running_[req->id] = req;
            batch.prefill_request_ids.push_back(req->id);
            token_budget -= prefill_tokens;
        }
        
        return batch;
    }
    
    // Mark request as having generated one more token
    void on_token_generated(int request_id, int token) {
        auto it = running_.find(request_id);
        if (it == running_.end()) return;
        
        auto& req = it->second;
        req->generated_tokens.push_back(token);
        req->tokens_generated++;
        
        // Check completion
        if (req->is_done() || token == EOS_TOKEN_ID) {
            req->status = RequestStatus::FINISHED;
            finished_.push_back(req);
            running_.erase(it);
        }
    }
    
    std::vector<std::shared_ptr<Request>> pop_finished() {
        auto result = finished_;
        finished_.clear();
        return result;
    }
    
    int queue_size() const { return (int)waiting_queue_.size(); }
    int running_size() const { return (int)running_.size(); }

private:
    static constexpr int EOS_TOKEN_ID = 2;
    
    // Priority queue: higher priority → processed first
    struct PriorityCompare {
        bool operator()(const std::shared_ptr<Request>& a,
                        const std::shared_ptr<Request>& b) const {
            // Lower priority value = lower priority (pop highest)
            if (a->priority != b->priority)
                return a->priority < b->priority;
            return a->arrival_time > b->arrival_time; // earlier arrival = higher priority
        }
    };
    
    std::priority_queue<
        std::shared_ptr<Request>,
        std::vector<std::shared_ptr<Request>>,
        PriorityCompare
    > waiting_queue_;
    
    std::unordered_map<int, std::shared_ptr<Request>> running_;
    std::vector<std::shared_ptr<Request>> finished_;
};
```

---

## I.4 Tokenizer Interface Pattern

```cpp
// include/tokenizer.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>  // nlohmann/json header-only library

// Minimal BPE tokenizer interface following Hugging Face tokenizer format.
// This is a simplified example — production use: link against tokenizers-cpp.

class SimpleTokenizer {
public:
    // Token types
    static constexpr int UNK_TOKEN = 0;
    static constexpr int BOS_TOKEN = 1;
    static constexpr int EOS_TOKEN = 2;
    
    // Load from Hugging Face tokenizer.json
    explicit SimpleTokenizer(const std::string& tokenizer_path) {
        std::ifstream f(tokenizer_path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open tokenizer: " + tokenizer_path);
        }
        
        nlohmann::json j;
        f >> j;
        
        // Load vocab
        auto& vocab = j["model"]["vocab"];
        for (auto& [token, id] : vocab.items()) {
            token_to_id_[token] = id.get<int>();
            id_to_token_[id.get<int>()] = token;
        }
        
        vocab_size_ = (int)token_to_id_.size();
    }
    
    // Encode text to token IDs
    std::vector<int> encode(const std::string& text, bool add_bos = false) const {
        std::vector<int> tokens;
        if (add_bos) tokens.push_back(BOS_TOKEN);
        
        // Simplified: character-level tokenization as placeholder
        // Real implementation would use BPE merge rules
        size_t pos = 0;
        while (pos < text.size()) {
            // Try longest match first (simplified)
            bool found = false;
            for (int len = std::min((int)text.size() - (int)pos, 20); len > 0; len--) {
                std::string substr = text.substr(pos, len);
                auto it = token_to_id_.find(substr);
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                    pos += len;
                    found = true;
                    break;
                }
            }
            if (!found) {
                tokens.push_back(UNK_TOKEN);
                pos++;
            }
        }
        return tokens;
    }
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& token_ids,
                       bool skip_special = true) const {
        std::string result;
        for (int id : token_ids) {
            if (skip_special && id <= 2) continue;  // skip BOS/EOS/UNK
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
                result += it->second;
            }
        }
        return result;
    }
    
    int vocab_size() const { return vocab_size_; }
    
    // Apply chat template (simplified ChatML format)
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages
    ) const {
        std::string result;
        for (auto& [role, content] : messages) {
            result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
        result += "<|im_start|>assistant\n";
        return result;
    }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    int vocab_size_ = 0;
};
```

---

## I.5 Metric Collection Pattern

```cpp
// include/metrics.h
#pragma once
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <mutex>
#include <string>
#include <sstream>

// Lock-free counter for high-frequency metrics
class AtomicCounter {
public:
    void increment(int64_t n = 1) { value_.fetch_add(n, std::memory_order_relaxed); }
    int64_t load() const { return value_.load(std::memory_order_relaxed); }
    void reset() { value_.store(0, std::memory_order_relaxed); }
private:
    std::atomic<int64_t> value_{0};
};

// Histogram for latency tracking
class LatencyHistogram {
public:
    void record(double ms) {
        std::lock_guard<std::mutex> lock(mu_);
        samples_.push_back(ms);
    }
    
    struct Stats {
        double mean, p50, p95, p99, max, min;
        size_t count;
    };
    
    Stats compute() {
        std::lock_guard<std::mutex> lock(mu_);
        if (samples_.empty()) return {0,0,0,0,0,0,0};
        
        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        
        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        size_t n = sorted.size();
        
        return {
            .mean = sum / n,
            .p50  = sorted[n * 50 / 100],
            .p95  = sorted[n * 95 / 100],
            .p99  = sorted[n * 99 / 100],
            .max  = sorted.back(),
            .min  = sorted.front(),
            .count = n,
        };
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        samples_.clear();
    }

private:
    std::mutex mu_;
    std::vector<double> samples_;
};

// RAII timer for latency measurement
class ScopedTimer {
public:
    explicit ScopedTimer(LatencyHistogram& histogram)
        : histogram_(histogram)
        , start_(std::chrono::steady_clock::now()) {}
    
    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        histogram_.record(ms);
    }

private:
    LatencyHistogram& histogram_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

// Prometheus-compatible metric exporter
class MetricsExporter {
public:
    struct EngineMetrics {
        AtomicCounter requests_total;
        AtomicCounter tokens_generated;
        AtomicCounter cache_hits;
        AtomicCounter cache_misses;
        LatencyHistogram ttft_ms;
        LatencyHistogram itl_ms;
    };
    
    // Export metrics in Prometheus text format
    static std::string to_prometheus(const EngineMetrics& m) {
        auto ttft = m.ttft_ms.compute();
        auto itl  = m.itl_ms.compute();
        
        std::ostringstream oss;
        oss << "# HELP vllm_requests_total Total requests processed\n";
        oss << "# TYPE vllm_requests_total counter\n";
        oss << "vllm_requests_total " << m.requests_total.load() << "\n\n";
        
        oss << "# HELP vllm_tokens_generated_total Total output tokens\n";
        oss << "# TYPE vllm_tokens_generated_total counter\n";
        oss << "vllm_tokens_generated_total " << m.tokens_generated.load() << "\n\n";
        
        oss << "# HELP vllm_cache_hit_rate KV cache hit rate\n";
        oss << "# TYPE vllm_cache_hit_rate gauge\n";
        int64_t hits = m.cache_hits.load(), misses = m.cache_misses.load();
        double hit_rate = (hits + misses > 0) ? (double)hits / (hits + misses) : 0.0;
        oss << "vllm_cache_hit_rate " << hit_rate << "\n\n";
        
        oss << "# HELP vllm_ttft_seconds Time to first token\n";
        oss << "# TYPE vllm_ttft_seconds summary\n";
        oss << "vllm_ttft_seconds{quantile=\"0.5\"} " << ttft.p50 / 1000.0 << "\n";
        oss << "vllm_ttft_seconds{quantile=\"0.95\"} " << ttft.p95 / 1000.0 << "\n";
        oss << "vllm_ttft_seconds{quantile=\"0.99\"} " << ttft.p99 / 1000.0 << "\n";
        oss << "vllm_ttft_seconds_count " << ttft.count << "\n\n";
        
        return oss.str();
    }
};
```

---

## I.6 Thread Pool Pattern for Parallel Requests

```cpp
// include/thread_pool.h
#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; i++) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mu_);
                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    
    // Submit a task and get a future for the result
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> {
        using RetT = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<RetT()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<RetT> fut = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mu_);
            if (stop_) throw std::runtime_error("ThreadPool stopped");
            tasks_.emplace([task]{ (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    
    size_t size() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// Example usage: parallel embedding computation
void parallel_embed_example() {
    ThreadPool pool(4);  // 4 worker threads
    
    std::vector<std::string> texts = {"Hello", "World", "AI", "Inference"};
    std::vector<std::future<std::vector<float>>> futures;
    
    for (const auto& text : texts) {
        futures.push_back(pool.submit([text]() -> std::vector<float> {
            // Simulate embedding computation
            std::vector<float> emb(128, 0.0f);
            for (int i = 0; i < 128; i++) {
                emb[i] = (float)(text.size() * i) / 1000.0f;
            }
            return emb;
        }));
    }
    
    // Collect results
    for (size_t i = 0; i < futures.size(); i++) {
        auto emb = futures[i].get();
        // use emb...
    }
}
```

---

## I.7 Compilation and Linking Patterns

### I.7.1 Compiling Mixed C++/CUDA

```makefile
# Compile CUDA kernel
nvcc -O2 -std=c++17 -arch=sm_90 \
     -c kernels/attention.cu -o attention.o

# Compile C++ code linking against CUDA kernel
g++ -O3 -std=c++17 \
    -I/usr/local/cuda/include \
    src/main.cpp attention.o \
    -L/usr/local/cuda/lib64 -lcudart -lcublas \
    -o inference_server
```

### I.7.2 Python Binding with pybind11

```cpp
// python_bindings.cpp — Expose C++ scheduler to Python
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "scheduler.h"

namespace py = pybind11;

PYBIND11_MODULE(inference_core, m) {
    m.doc() = "C++ inference core bindings";
    
    py::class_<Request, std::shared_ptr<Request>>(m, "Request")
        .def(py::init<int, std::vector<int>, int>())
        .def_readwrite("id", &Request::id)
        .def_readwrite("tokens_generated", &Request::tokens_generated)
        .def_readwrite("generated_tokens", &Request::generated_tokens)
        .def("is_done", &Request::is_done)
        .def("total_tokens", &Request::total_tokens);
    
    py::class_<ScheduleBatch>(m, "ScheduleBatch")
        .def_readwrite("prefill_request_ids", &ScheduleBatch::prefill_request_ids)
        .def_readwrite("decode_request_ids", &ScheduleBatch::decode_request_ids);
    
    py::class_<ContinuousBatchScheduler>(m, "ContinuousBatchScheduler")
        .def(py::init<int, int>(),
             py::arg("max_batch_tokens") = 2048,
             py::arg("max_sequences") = 64)
        .def("add_request", &ContinuousBatchScheduler::add_request)
        .def("schedule", &ContinuousBatchScheduler::schedule)
        .def("on_token_generated", &ContinuousBatchScheduler::on_token_generated)
        .def("pop_finished", &ContinuousBatchScheduler::pop_finished)
        .def("queue_size", &ContinuousBatchScheduler::queue_size)
        .def("running_size", &ContinuousBatchScheduler::running_size);
}
```

```bash
# Build Python extension
g++ -O3 -std=c++17 -fPIC \
    -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
    -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
    python_bindings.cpp src/scheduler.cpp \
    -shared -o inference_core$(python3-config --extension-suffix)

# Use in Python
python3 -c "
import inference_core
sched = inference_core.ContinuousBatchScheduler(2048, 64)
req = inference_core.Request(0, [1, 2, 3, 4, 5], 100)
sched.add_request(req)
batch = sched.schedule()
print('Prefill:', batch.prefill_request_ids)
"
```

---

## I.8 Common Pitfalls and Solutions

```
Issue                             | Cause                    | Fix
──────────────────────────────────────────────────────────────────────────
Undefined behavior in KV indexing | Off-by-one in block math | Use assert() for all index bounds
Data race in metrics              | Multiple threads writing  | Use std::atomic or mutex
Memory leak in block manager      | ref_count logic bug       | Add destructor validation
CUDA/CPU result mismatch          | Different float precision | Use tolerance comparison in tests
Slow Python bindings              | Copying large vectors     | Use py::array_t for buffer protocol
Linker error: undefined CUDA syms | Missing -lcudart          | Add to LDFLAGS
```

```cpp
// Testing pattern: assert with tolerance
template<typename T>
void assert_near(T a, T b, T tol, const std::string& msg = "") {
    if (std::abs(a - b) > tol) {
        throw std::runtime_error(
            "assert_near failed: |" + std::to_string(a) + " - " + 
            std::to_string(b) + "| = " + std::to_string(std::abs(a-b)) +
            " > " + std::to_string(tol) + 
            (msg.empty() ? "" : " [" + msg + "]")
        );
    }
}
```
