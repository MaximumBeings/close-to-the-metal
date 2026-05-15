# Chapter 6: PagedAttention and the Block Manager — Companion Code

## Python — `paged_attention_demo.py`

```python
# paged_attention_demo.py
# Chapter 6 — PagedAttention and the Block Manager
#
# Simulates:
#   1. Block allocator (GPU + CPU)
#   2. Block table per sequence
#   3. Prefill + decode with on-demand block allocation
#   4. Copy-on-write for beam search
#   5. Prefix caching with LRU eviction
#   6. GPU → CPU swapping
#
# Requirements:
#   pip install numpy   (only needed for Section 7 attention kernel)
#
# Run:
#   python paged_attention_demo.py

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Core data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicalBlock:
    """
    Represents one physical KV-cache block.

    In a real GPU implementation block_id is an index into a pre-allocated
    tensor of shape [num_blocks, 2, n_layers, block_size, n_heads, d_head].
    """
    block_id:           int
    device:             str          # "gpu" | "cpu"
    ref_count:          int  = 0
    block_hash:         Optional[int] = None
    num_hashed_tokens:  int  = 0

    def __repr__(self) -> str:
        return (f"Block(id={self.block_id}, dev={self.device}, "
                f"ref={self.ref_count}, hash={self.block_hash})")


class BlockAllocatorError(Exception):
    pass


class BlockAllocator:
    """
    Pool of physical blocks on a single device.

    Supports:
    - Basic alloc / free
    - Prefix-cache lookup (hash → block, LRU eviction order)
    """

    def __init__(self, device: str, num_blocks: int, block_size: int) -> None:
        self.device     = device
        self.block_size = block_size
        self._num_total = num_blocks
        # Free list (as a stack for O(1) alloc/free)
        self._free: List[PhysicalBlock] = [
            PhysicalBlock(i, device) for i in range(num_blocks)
        ]
        # Prefix-cache: hash → block (OrderedDict preserves LRU order)
        self._cache: OrderedDict[int, PhysicalBlock] = OrderedDict()

    # ── alloc / free ──────────────────────────────────────────────────────

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalBlock:
        """
        Allocate one physical block.
        If the free list is empty, evict the LRU cached block.
        """
        if not self._free:
            self._evict_lru()
        if not self._free:
            raise BlockAllocatorError(
                f"No free blocks on {self.device} (total={self._num_total})"
            )
        blk = self._free.pop()
        blk.ref_count         = 1
        blk.block_hash        = block_hash
        blk.num_hashed_tokens = num_hashed_tokens
        if block_hash is not None:
            self._cache[block_hash] = blk
            self._cache.move_to_end(block_hash)  # mark as most-recently-used
        return blk

    def free(self, blk: PhysicalBlock) -> None:
        """Decrement ref count; truly free only when count reaches 0."""
        assert blk.ref_count > 0, f"Double-free of {blk}"
        blk.ref_count -= 1
        if blk.ref_count == 0:
            if blk.block_hash is not None:
                # Keep in cache for potential prefix-cache reuse.
                # It will be evicted if the free list runs dry.
                pass
            else:
                self._free.append(blk)

    # ── prefix cache ──────────────────────────────────────────────────────

    def get_cached(self, block_hash: int) -> Optional[PhysicalBlock]:
        """
        Look up block by hash.  On hit, increment ref_count and mark MRU.
        """
        if block_hash in self._cache:
            blk = self._cache[block_hash]
            blk.ref_count += 1
            self._cache.move_to_end(block_hash)
            return blk
        return None

    def _evict_lru(self) -> None:
        """Evict the LRU unreferenced cached block."""
        for h, blk in self._cache.items():   # LRU = first item
            if blk.ref_count == 0:
                del self._cache[h]
                blk.block_hash        = None
                blk.num_hashed_tokens = 0
                self._free.append(blk)
                return
        raise BlockAllocatorError(
            f"All {self._num_total} blocks on {self.device} are referenced"
        )

    @property
    def num_free_blocks(self) -> int:
        """Free list + unreferenced cached blocks."""
        cached_free = sum(1 for b in self._cache.values() if b.ref_count == 0)
        return len(self._free) + cached_free

    @property
    def num_used_blocks(self) -> int:
        return self._num_total - self.num_free_blocks

    def __repr__(self) -> str:
        return (f"BlockAllocator(device={self.device}, "
                f"free={self.num_free_blocks}/{self._num_total}, "
                f"cached={len(self._cache)})")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Block table
# ─────────────────────────────────────────────────────────────────────────────

def _chain_hash(token_ids: List[int], prev_hash: int) -> int:
    """
    Hash a block's token IDs chained with the previous block's hash.
    This ensures two blocks with the same content at different positions
    in the sequence get different hashes.
    """
    raw = f"{prev_hash}:{token_ids}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


class BlockTable:
    """
    Per-sequence mapping from logical block indices to physical blocks.

    Logical block index = token_position // block_size
    Within-block offset  = token_position %  block_size
    """

    def __init__(self, block_size: int, allocator: BlockAllocator) -> None:
        self._blocks:     List[PhysicalBlock] = []
        self.block_size:  int                 = block_size
        self._allocator:  BlockAllocator      = allocator
        self._num_tokens: int                 = 0

    # ── prefill ───────────────────────────────────────────────────────────

    def allocate(self, token_ids: List[int], prev_hash: int = 0) -> int:
        """
        Allocate blocks for a full token list, checking prefix cache first.
        Returns the number of cache hits (blocks reused without recompute).
        """
        n            = len(token_ids)
        num_full     = n // self.block_size
        remainder    = n %  self.block_size
        hits         = 0

        for b in range(num_full):
            chunk = token_ids[b * self.block_size : (b + 1) * self.block_size]
            h     = _chain_hash(chunk, prev_hash)
            cached = self._allocator.get_cached(h)
            if cached is not None:
                self._blocks.append(cached)
                hits += 1
            else:
                blk = self._allocator.allocate(
                    block_hash=h,
                    num_hashed_tokens=(b + 1) * self.block_size,
                )
                self._blocks.append(blk)
            prev_hash = h

        if remainder:
            blk = self._allocator.allocate()   # partial block: no hash
            self._blocks.append(blk)

        self._num_tokens = n
        return hits

    # ── decode ────────────────────────────────────────────────────────────

    def append_slot(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Reserve space for one more token.

        Returns
        -------
        (None,     None)    — appended to existing block, no copy
        (new_id,   None)    — new block allocated
        (old_id,   new_id)  — copy-on-write performed
        """
        if not self._blocks:
            raise RuntimeError("BlockTable is empty; call allocate() first")

        tokens_in_last = self._num_tokens % self.block_size
        if tokens_in_last == 0:
            # Current last block is full; need a new one.
            new_blk = self._allocator.allocate()
            self._blocks.append(new_blk)
            self._num_tokens += 1
            return (new_blk.block_id, None)

        # Space in the last block.
        last = self._blocks[-1]
        if last.ref_count > 1:
            return self._cow(last)

        self._num_tokens += 1
        return (None, None)

    def _cow(self, blk: PhysicalBlock) -> Tuple[int, int]:
        """Copy-on-write: replace shared last block with a private copy."""
        new_blk = self._allocator.allocate()
        old_id  = blk.block_id
        self._allocator.free(blk)     # decrement ref on shared block
        self._blocks[-1] = new_blk
        self._num_tokens += 1
        return (old_id, new_blk.block_id)

    # ── beam search fork ──────────────────────────────────────────────────

    def fork(self) -> "BlockTable":
        """
        Create a child BlockTable sharing all current blocks.
        Increments ref_count on every shared block.
        """
        child              = BlockTable(self.block_size, self._allocator)
        child._blocks      = list(self._blocks)
        child._num_tokens  = self._num_tokens
        for blk in child._blocks:
            blk.ref_count += 1
        return child

    # ── teardown ──────────────────────────────────────────────────────────

    def free_all(self) -> None:
        for blk in self._blocks:
            self._allocator.free(blk)
        self._blocks.clear()
        self._num_tokens = 0

    @property
    def physical_block_ids(self) -> List[int]:
        return [b.block_id for b in self._blocks]

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def __repr__(self) -> str:
        return (f"BlockTable(tokens={self._num_tokens}, "
                f"logical_blocks={len(self._blocks)}, "
                f"phys={self.physical_block_ids})")


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Swap helpers
# ─────────────────────────────────────────────────────────────────────────────

def swap_out(bt: BlockTable,
             gpu_alloc: BlockAllocator,
             cpu_alloc: BlockAllocator) -> Dict[int, int]:
    """
    Move all blocks of a BlockTable from GPU to CPU.
    Returns gpu_id → cpu_id mapping.
    In a real implementation this would issue CUDA memcpy D→H for each block.
    """
    mapping: Dict[int, int] = {}
    for i, gpu_blk in enumerate(bt._blocks):
        cpu_blk = cpu_alloc.allocate()
        print(f"    memcpy D→H: GPU_block_{gpu_blk.block_id} "
              f"→ CPU_block_{cpu_blk.block_id}")
        mapping[gpu_blk.block_id] = cpu_blk.block_id
        gpu_alloc.free(gpu_blk)
        bt._blocks[i] = cpu_blk
    return mapping


def swap_in(bt: BlockTable,
            gpu_alloc: BlockAllocator,
            cpu_alloc: BlockAllocator) -> Dict[int, int]:
    """
    Move all blocks from CPU back to GPU.
    Returns cpu_id → gpu_id mapping.
    """
    mapping: Dict[int, int] = {}
    for i, cpu_blk in enumerate(bt._blocks):
        gpu_blk = gpu_alloc.allocate()
        print(f"    memcpy H→D: CPU_block_{cpu_blk.block_id} "
              f"→ GPU_block_{gpu_blk.block_id}")
        mapping[cpu_blk.block_id] = gpu_blk.block_id
        cpu_alloc.free(cpu_blk)
        bt._blocks[i] = gpu_blk
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Paged attention kernel (toy NumPy implementation)
# ─────────────────────────────────────────────────────────────────────────────

def paged_attention_single_query(
    q:           np.ndarray,           # [d_k]  — single query vector
    kv_store:    np.ndarray,           # [num_blocks, 2, block_size, d_k]
    block_table: List[int],            # logical → physical block ids
    num_tokens:  int,                  # how many KV tokens are valid
    block_size:  int,
) -> np.ndarray:
    """
    Compute attention output for a single query over a paged KV cache.

    This is the decode-time kernel:  one query attends to all past KV tokens
    stored in non-contiguous physical blocks.

    Uses online softmax (Chapter 5 §5.3) to accumulate over blocks without
    ever materialising the full attention score vector.

    Returns output vector [d_k].
    """
    d_k   = q.shape[0]
    scale = 1.0 / np.sqrt(d_k)

    # Online softmax state
    m_global = -np.inf       # running max
    l_global = 0.0           # running sum of exp(s - m)
    o_global = np.zeros(d_k)

    tokens_processed = 0
    for logical_idx, phys_id in enumerate(block_table):
        tokens_in_block = min(block_size, num_tokens - tokens_processed)
        if tokens_in_block <= 0:
            break

        K_block = kv_store[phys_id, 0, :tokens_in_block, :]   # [t, d_k]
        V_block = kv_store[phys_id, 1, :tokens_in_block, :]   # [t, d_k]

        # Scores for this block
        s = (K_block @ q) * scale                              # [t]

        # Online softmax merge
        m_block = s.max()
        l_block = np.exp(s - m_block).sum()
        o_block = np.exp(s - m_block) @ V_block                # [d_k]

        m_new = max(m_global, m_block)
        l_global = (np.exp(m_global - m_new) * l_global
                    + np.exp(m_block  - m_new) * l_block)
        o_global = (np.exp(m_global - m_new) * o_global
                    + np.exp(m_block  - m_new) * o_block)
        m_global = m_new

        tokens_processed += tokens_in_block

    if l_global > 0:
        o_global /= l_global
    return o_global


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Basic allocation
# ═════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 4
NUM_BLOCKS = 12

print("=" * 60)
print("SECTION 1: Basic Allocation (block_size=4, pool=12)")
print("=" * 60)

gpu_alloc = BlockAllocator("gpu", NUM_BLOCKS, BLOCK_SIZE)
print(f"\nInitial: {gpu_alloc}\n")

# r1: prefill 7 tokens
print("r1 prefill (7 tokens):")
bt_r1 = BlockTable(BLOCK_SIZE, gpu_alloc)
hits = bt_r1.allocate(list(range(7)))
print(f"  {bt_r1}  (cache hits={hits})")

# r2: prefill 5 tokens
print("\nr2 prefill (5 tokens):")
bt_r2 = BlockTable(BLOCK_SIZE, gpu_alloc)
bt_r2.allocate(list(range(5)))
print(f"  {bt_r2}")

print(f"\nGPU state after two prefills: {gpu_alloc}")

# Decode: r1 generates 5 more tokens
print("\nr1 decode (5 tokens):")
for step in range(5):
    result = bt_r1.append_slot()
    if result != (None, None):
        kind = "new block" if result[1] is None else "CoW"
        print(f"  step {step+1}: {kind} event {result}")
    else:
        print(f"  step {step+1}: in-place append")
print(f"  {bt_r1}")

# r2 completes
print(f"\nr2 completes — freeing blocks {bt_r2.physical_block_ids}:")
bt_r2.free_all()
print(f"  GPU state: {gpu_alloc}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Copy-on-Write (beam search, B=3)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2: Copy-on-Write — Beam Search (B=3)")
print("=" * 60)

gpu2 = BlockAllocator("gpu", 20, BLOCK_SIZE)

print("\nShared prefill (12 tokens → 3 full blocks):")
parent = BlockTable(BLOCK_SIZE, gpu2)
parent.allocate(list(range(12)))
print(f"  parent: {parent}")

print("\nForking 3 beams:")
beams = [parent.fork() for _ in range(3)]
for i, b in enumerate(beams):
    ref_counts = [blk.ref_count for blk in b._blocks]
    print(f"  beam_{i}: {b}  ref_counts={ref_counts}")

print("\nDecode step 1:")
for i, beam in enumerate(beams):
    result = beam.append_slot()
    kind = "in-place" if result == (None, None) else \
           ("new-block" if result[1] is None else "CoW")
    print(f"  beam_{i}: {kind} → phys_ids={beam.physical_block_ids}")

print("\nRef counts on originally-shared blocks after step 1:")
for blk in parent._blocks:
    print(f"  phys_{blk.block_id}: ref_count={blk.ref_count}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Prefix caching
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3: Prefix Caching")
print("=" * 60)

gpu3 = BlockAllocator("gpu", 30, BLOCK_SIZE)

SYSTEM_PROMPT_TOKENS = list(range(100, 116))   # 16 tokens → 4 full blocks

print(f"\nSystem prompt: {len(SYSTEM_PROMPT_TOKENS)} tokens "
      f"({len(SYSTEM_PROMPT_TOKENS)//BLOCK_SIZE} full blocks)")

total_hits = 0
total_blocks = 0

for req_id in range(5):
    user_tokens = list(range(200 + req_id * 10, 204 + req_id * 10))
    all_tokens  = SYSTEM_PROMPT_TOKENS + user_tokens
    bt = BlockTable(BLOCK_SIZE, gpu3)
    hits = bt.allocate(all_tokens)
    total_hits   += hits
    total_blocks += len(bt._blocks)
    print(f"  req_{req_id}: {len(all_tokens)} tokens, "
          f"blocks={bt.physical_block_ids}, "
          f"cache_hits={hits}/{len(SYSTEM_PROMPT_TOKENS)//BLOCK_SIZE} prefix blocks")

print(f"\nOverall prefix-cache hit rate: "
      f"{total_hits}/{total_blocks} blocks = "
      f"{total_hits/total_blocks*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: GPU ↔ CPU Swap
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 4: GPU ↔ CPU Swap")
print("=" * 60)

GPU_BLOCKS_4 = 8
CPU_BLOCKS_4 = 16

gpu4 = BlockAllocator("gpu", GPU_BLOCKS_4, BLOCK_SIZE)
cpu4 = BlockAllocator("cpu", CPU_BLOCKS_4, BLOCK_SIZE)

print(f"\nAllocate seq_a (8 tokens = 2 blocks) and seq_b (8 tokens = 2 blocks):")
bt_a4 = BlockTable(BLOCK_SIZE, gpu4); bt_a4.allocate(list(range(8)))
bt_b4 = BlockTable(BLOCK_SIZE, gpu4); bt_b4.allocate(list(range(8, 16)))
print(f"  seq_a: {bt_a4}")
print(f"  seq_b: {bt_b4}")
print(f"  {gpu4}")

print(f"\nGPU pool at capacity — swap out seq_b to CPU:")
swap_out(bt_b4, gpu4, cpu4)
print(f"  seq_b now on CPU: {bt_b4}")
print(f"  {gpu4}")

print(f"\nAdmit seq_c (8 tokens) — now possible:")
bt_c4 = BlockTable(BLOCK_SIZE, gpu4); bt_c4.allocate(list(range(16, 24)))
print(f"  seq_c: {bt_c4}")

print(f"\nseq_c finishes — swap seq_b back to GPU:")
bt_c4.free_all()
swap_in(bt_b4, gpu4, cpu4)
print(f"  seq_b back on GPU: {bt_b4}")
print(f"  {gpu4}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Paged attention kernel correctness check
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 5: Paged vs Standard Attention Kernel")
print("=" * 60)

np.random.seed(0)
BSIZE  = 4
D_K    = 8
N_PHYS = 6
N_TOK  = 10

# Toy KV store: shape [N_PHYS, 2, BSIZE, D_K]
kv_store = np.random.randn(N_PHYS, 2, BSIZE, D_K).astype(np.float32)

# A sequence of 10 tokens scattered across 3 blocks: phys 2, 5, 0
block_table_ids = [2, 5, 0]   # logical 0→phys_2, 1→phys_5, 2→phys_0

# Reconstruct contiguous K, V for reference attention
K_ref = np.concatenate([
    kv_store[2, 0, :4],    # first  4 tokens
    kv_store[5, 0, :4],    # next   4 tokens
    kv_store[0, 0, :2],    # last   2 tokens
], axis=0)   # [10, D_K]

V_ref = np.concatenate([
    kv_store[2, 1, :4],
    kv_store[5, 1, :4],
    kv_store[0, 1, :2],
], axis=0)   # [10, D_K]

q = np.random.randn(D_K).astype(np.float32)

# Reference: standard softmax attention
scores = (K_ref @ q) / np.sqrt(D_K)
weights = np.exp(scores - scores.max())
weights /= weights.sum()
o_ref = weights @ V_ref

# Paged kernel
o_paged = paged_attention_single_query(
    q, kv_store, block_table_ids, N_TOK, BSIZE
)

err = np.abs(o_ref - o_paged).max()
print(f"\n  Max absolute error (paged vs standard): {err:.2e}")
assert err < 1e-5, f"Paged attention mismatch: {err}"
print("  ✓ Paged attention matches standard attention exactly.")

print("\nDone.")

```

## C++ — `paged_attention_demo.cpp`

```cpp
// paged_attention_demo.cpp
// Chapter 6 — PagedAttention and the Block Manager
//
// Implements (no third-party libraries beyond std):
//   1. PhysicalBlock / BlockAllocator with LRU prefix cache
//   2. BlockTable with on-demand allocation, CoW, fork
//   3. GPU ↔ CPU swap simulation
//   4. Paged attention single-query kernel (online softmax)
//   5. Correctness check: paged vs contiguous standard attention
//
// Build:
//   g++ -std=c++17 -O2 paged_attention_demo.cpp -o paged_attention_demo
//
// Run:
//   ./paged_attention_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// PhysicalBlock
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalBlock {
    int         block_id;
    std::string device;      // "gpu" | "cpu"
    int         ref_count         = 0;
    int         block_hash        = -1;   // -1 = no hash
    int         num_hashed_tokens = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// BlockAllocator
// ─────────────────────────────────────────────────────────────────────────────

class BlockAllocator {
public:
    BlockAllocator(const std::string& device, int num_blocks, int block_size)
        : device_(device), block_size_(block_size), num_total_(num_blocks)
    {
        storage_.resize(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            storage_[i].block_id = i;
            storage_[i].device   = device;
            free_.push_back(&storage_[i]);
        }
    }

    // Allocate one block (evict LRU cached block if pool empty)
    PhysicalBlock* allocate(int block_hash = -1, int num_hashed_tokens = 0) {
        if (free_.empty())
            evict_lru();

        PhysicalBlock* blk = free_.back();
        free_.pop_back();
        blk->ref_count         = 1;
        blk->block_hash        = block_hash;
        blk->num_hashed_tokens = num_hashed_tokens;

        if (block_hash != -1) {
            cache_order_.push_front(block_hash);
            cache_[block_hash] = {blk, cache_order_.begin()};
        }
        return blk;
    }

    // Decrement ref; only truly freed when ref_count reaches 0
    void free(PhysicalBlock* blk) {
        assert(blk->ref_count > 0);
        blk->ref_count--;
        if (blk->ref_count == 0 && blk->block_hash == -1) {
            free_.push_back(blk);
        }
        // If ref_count==0 but block is cached, it stays in cache for reuse.
    }

    // Prefix cache lookup
    PhysicalBlock* get_cached(int block_hash) {
        auto it = cache_.find(block_hash);
        if (it == cache_.end()) return nullptr;
        PhysicalBlock* blk = it->second.first;
        blk->ref_count++;
        // Move to front (MRU)
        cache_order_.splice(cache_order_.begin(),
                            cache_order_, it->second.second);
        it->second.second = cache_order_.begin();
        return blk;
    }

    int num_free_blocks() const {
        int cached_free = 0;
        for (auto& [h, p] : cache_)
            if (p.first->ref_count == 0) ++cached_free;
        return (int)free_.size() + cached_free;
    }

    const std::string& device() const { return device_; }
    int block_size() const { return block_size_; }

private:
    void evict_lru() {
        // LRU = back of cache_order_
        for (auto it = cache_order_.rbegin(); it != cache_order_.rend(); ++it) {
            auto cit = cache_.find(*it);
            PhysicalBlock* blk = cit->second.first;
            if (blk->ref_count == 0) {
                blk->block_hash        = -1;
                blk->num_hashed_tokens = 0;
                free_.push_back(blk);
                cache_.erase(cit);
                cache_order_.erase(std::next(it).base());
                return;
            }
        }
        fprintf(stderr, "BlockAllocator: all blocks referenced, cannot evict\n");
        std::exit(1);
    }

    std::string              device_;
    int                      block_size_;
    int                      num_total_;
    std::vector<PhysicalBlock> storage_;
    std::vector<PhysicalBlock*> free_;

    // LRU cache: hash → (block*, iterator into cache_order_)
    std::list<int>   cache_order_;
    std::unordered_map<int, std::pair<PhysicalBlock*, std::list<int>::iterator>> cache_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: chain hash  (matches Python implementation)
// ─────────────────────────────────────────────────────────────────────────────

static int chain_hash(const std::vector<int>& token_ids, int prev_hash) {
    // Simple polynomial hash (not cryptographic; for demo only)
    unsigned h = (unsigned)prev_hash;
    for (int t : token_ids) {
        h = h * 1000003u ^ (unsigned)t;
    }
    return (int)(h & 0x7FFFFFFF);
}

// ─────────────────────────────────────────────────────────────────────────────
// BlockTable
// ─────────────────────────────────────────────────────────────────────────────

class BlockTable {
public:
    BlockTable(int block_size, BlockAllocator* alloc)
        : block_size_(block_size), alloc_(alloc), num_tokens_(0) {}

    // ── prefill ──────────────────────────────────────────────────────────
    int allocate(const std::vector<int>& token_ids, int prev_hash = 0) {
        int n         = (int)token_ids.size();
        int num_full  = n / block_size_;
        int remainder = n % block_size_;
        int hits      = 0;

        for (int b = 0; b < num_full; ++b) {
            std::vector<int> chunk(token_ids.begin() + b * block_size_,
                                   token_ids.begin() + (b + 1) * block_size_);
            int h = chain_hash(chunk, prev_hash);
            PhysicalBlock* cached = alloc_->get_cached(h);
            if (cached) {
                blocks_.push_back(cached);
                ++hits;
            } else {
                PhysicalBlock* blk = alloc_->allocate(h, (b + 1) * block_size_);
                blocks_.push_back(blk);
            }
            prev_hash = h;
        }
        if (remainder) {
            blocks_.push_back(alloc_->allocate());
        }
        num_tokens_ = n;
        return hits;
    }

    // ── decode: append one token ──────────────────────────────────────────
    // Returns: (-1, -1)   in-place append
    //          (new_id, -1) new block allocated
    //          (old_id, new_id) CoW performed
    std::pair<int,int> append_slot() {
        assert(!blocks_.empty());
        int tokens_in_last = num_tokens_ % block_size_;
        if (tokens_in_last == 0) {
            // Last block full — allocate new
            PhysicalBlock* blk = alloc_->allocate();
            blocks_.push_back(blk);
            ++num_tokens_;
            return {blk->block_id, -1};
        }
        PhysicalBlock* last = blocks_.back();
        if (last->ref_count > 1) {
            return cow(last);
        }
        ++num_tokens_;
        return {-1, -1};
    }

    // ── fork (for beam search) ────────────────────────────────────────────
    BlockTable fork() const {
        BlockTable child(block_size_, alloc_);
        child.blocks_     = blocks_;
        child.num_tokens_ = num_tokens_;
        for (auto* blk : child.blocks_) blk->ref_count++;
        return child;
    }

    void free_all() {
        for (auto* blk : blocks_) alloc_->free(blk);
        blocks_.clear();
        num_tokens_ = 0;
    }

    std::vector<int> physical_block_ids() const {
        std::vector<int> ids;
        for (auto* b : blocks_) ids.push_back(b->block_id);
        return ids;
    }

    int num_tokens() const { return num_tokens_; }

    // For inspecting ref counts
    const std::vector<PhysicalBlock*>& blocks() const { return blocks_; }

private:
    std::pair<int,int> cow(PhysicalBlock* old_blk) {
        PhysicalBlock* new_blk = alloc_->allocate();
        alloc_->free(old_blk);          // decrement ref on shared block
        blocks_.back() = new_blk;
        ++num_tokens_;
        return {old_blk->block_id, new_blk->block_id};
    }

    int             block_size_;
    BlockAllocator* alloc_;
    int             num_tokens_;
    std::vector<PhysicalBlock*> blocks_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Swap helpers
// ─────────────────────────────────────────────────────────────────────────────

void swap_out(BlockTable& bt, BlockAllocator& gpu, BlockAllocator& cpu) {
    // In a real system: CUDA memcpy D→H for each block
    auto ids = bt.physical_block_ids();
    // We rebuild the block table on CPU by re-allocating all blocks there
    // (simulation only — no actual data copy)
    bt.free_all();
    // Re-allocate on CPU (same number of tokens)
    // For simplicity we just print the mapping
    for (int gid : ids) {
        PhysicalBlock* cblk = cpu.allocate();
        printf("    memcpy D→H: GPU_block_%d → CPU_block_%d\n", gid, cblk->block_id);
    }
}

void swap_in(BlockTable& bt, BlockAllocator& gpu, BlockAllocator& cpu,
             const std::vector<int>& cpu_ids) {
    for (int cid : cpu_ids) {
        PhysicalBlock* gblk = gpu.allocate();
        printf("    memcpy H→D: CPU_block_%d → GPU_block_%d\n", cid, gblk->block_id);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Paged attention kernel (online softmax, single query)
// ─────────────────────────────────────────────────────────────────────────────

// kv_store[phys_id][k_or_v][slot][dim]
// shape: [num_phys_blocks][2][block_size][d_k]
using KVStore = std::vector<std::vector<std::array<std::vector<float>, 2>>>;

std::vector<float> paged_attention_single_query(
    const std::vector<float>& q,
    const KVStore& kv_store,
    const std::vector<int>& block_table,
    int num_tokens,
    int block_size)
{
    int d_k   = (int)q.size();
    float scale = 1.0f / std::sqrt((float)d_k);

    float m_global = -1e30f;
    float l_global =  0.0f;
    std::vector<float> o_global(d_k, 0.0f);

    int tokens_processed = 0;
    for (int phys_id : block_table) {
        int tokens_in_block = std::min(block_size,
                                       num_tokens - tokens_processed);
        if (tokens_in_block <= 0) break;

        // Compute scores for this block
        float m_block = -1e30f;
        std::vector<float> scores(tokens_in_block);
        for (int t = 0; t < tokens_in_block; ++t) {
            const auto& k_vec = kv_store[phys_id][t][0];  // K slot t
            float s = 0;
            for (int i = 0; i < d_k; ++i) s += k_vec[i] * q[i];
            scores[t] = s * scale;
            m_block   = std::max(m_block, scores[t]);
        }

        float l_block = 0.0f;
        std::vector<float> o_block(d_k, 0.0f);
        for (int t = 0; t < tokens_in_block; ++t) {
            float a = std::exp(scores[t] - m_block);
            l_block += a;
            const auto& v_vec = kv_store[phys_id][t][1];  // V slot t
            for (int i = 0; i < d_k; ++i) o_block[i] += a * v_vec[i];
        }

        // Merge into global state
        float m_new = std::max(m_global, m_block);
        float scale_old = std::exp(m_global - m_new);
        float scale_new = std::exp(m_block  - m_new);
        l_global = scale_old * l_global + scale_new * l_block;
        for (int i = 0; i < d_k; ++i)
            o_global[i] = scale_old * o_global[i] + scale_new * o_block[i];
        m_global = m_new;

        tokens_processed += tokens_in_block;
    }

    if (l_global > 0)
        for (auto& v : o_global) v /= l_global;
    return o_global;
}

// Standard contiguous attention (reference)
std::vector<float> standard_attention(
    const std::vector<float>& q,
    const std::vector<std::vector<float>>& K,
    const std::vector<std::vector<float>>& V)
{
    int n = (int)K.size(), d_k = (int)q.size();
    float scale = 1.0f / std::sqrt((float)d_k);

    std::vector<float> scores(n);
    for (int t = 0; t < n; ++t) {
        float s = 0;
        for (int i = 0; i < d_k; ++i) s += K[t][i] * q[i];
        scores[t] = s * scale;
    }
    float m = *std::max_element(scores.begin(), scores.end());
    float sum = 0;
    for (auto& s : scores) { s = std::exp(s - m); sum += s; }
    for (auto& s : scores) s /= sum;

    std::vector<float> out(d_k, 0.0f);
    for (int t = 0; t < n; ++t)
        for (int i = 0; i < d_k; ++i)
            out[i] += scores[t] * V[t][i];
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// print helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_ids(const std::vector<int>& ids) {
    printf("[");
    for (int i = 0; i < (int)ids.size(); ++i) {
        printf("%d", ids[i]);
        if (i + 1 < (int)ids.size()) printf(", ");
    }
    printf("]");
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {

    // ─────────────────────────────────────────────────────────────────────
    // SECTION 1: Basic Allocation
    // ─────────────────────────────────────────────────────────────────────
    printf("============================================================\n");
    printf("SECTION 1: Basic Allocation (block_size=4, pool=12)\n");
    printf("============================================================\n\n");

    const int BSIZE = 4;
    BlockAllocator gpu1("gpu", 12, BSIZE);

    printf("r1 prefill (7 tokens):\n");
    BlockTable bt_r1(BSIZE, &gpu1);
    std::vector<int> r1_tokens(7);
    std::iota(r1_tokens.begin(), r1_tokens.end(), 0);
    bt_r1.allocate(r1_tokens);
    auto ids_r1 = bt_r1.physical_block_ids();
    printf("  phys_ids="); print_ids(ids_r1); printf(", tokens=%d\n", bt_r1.num_tokens());

    printf("r2 prefill (5 tokens):\n");
    BlockTable bt_r2(BSIZE, &gpu1);
    std::vector<int> r2_tokens(5);
    std::iota(r2_tokens.begin(), r2_tokens.end(), 0);
    bt_r2.allocate(r2_tokens);
    auto ids_r2 = bt_r2.physical_block_ids();
    printf("  phys_ids="); print_ids(ids_r2); printf(", tokens=%d\n", bt_r2.num_tokens());

    printf("r1 decode (5 tokens):\n");
    for (int step = 0; step < 5; ++step) {
        auto [a, b] = bt_r1.append_slot();
        if (a == -1 && b == -1)
            printf("  step %d: in-place append\n", step + 1);
        else if (b == -1)
            printf("  step %d: new block allocated, id=%d\n", step + 1, a);
        else
            printf("  step %d: CoW %d → %d\n", step + 1, a, b);
    }
    ids_r1 = bt_r1.physical_block_ids();
    printf("  after decode: phys_ids="); print_ids(ids_r1);
    printf(", tokens=%d\n", bt_r1.num_tokens());

    printf("r2 completes — free blocks "); print_ids(ids_r2); printf("\n");
    bt_r2.free_all();
    printf("  GPU free blocks: %d\n\n", gpu1.num_free_blocks());


    // ─────────────────────────────────────────────────────────────────────
    // SECTION 2: Copy-on-Write (beam search, B=3)
    // ─────────────────────────────────────────────────────────────────────
    printf("============================================================\n");
    printf("SECTION 2: Copy-on-Write — Beam Search (B=3)\n");
    printf("============================================================\n\n");

    BlockAllocator gpu2("gpu", 20, BSIZE);

    printf("Shared prefill (12 tokens → 3 full blocks):\n");
    BlockTable parent(BSIZE, &gpu2);
    std::vector<int> parent_tok(12); std::iota(parent_tok.begin(), parent_tok.end(), 0);
    parent.allocate(parent_tok);
    auto parent_ids = parent.physical_block_ids();
    printf("  parent: phys_ids="); print_ids(parent_ids); printf("\n");

    printf("Forking 3 beams:\n");
    std::vector<BlockTable> beams;
    for (int i = 0; i < 3; ++i) {
        beams.push_back(parent.fork());
        auto bid = beams.back().physical_block_ids();
        printf("  beam_%d: phys_ids=", i); print_ids(bid); printf("\n");
    }

    printf("Ref counts on shared blocks: ");
    for (auto* blk : parent.blocks())
        printf("phys_%d(ref=%d) ", blk->block_id, blk->ref_count);
    printf("\n");

    printf("Decode step 1:\n");
    for (int i = 0; i < 3; ++i) {
        auto [a, b] = beams[i].append_slot();
        auto bid = beams[i].physical_block_ids();
        if (a == -1 && b == -1)
            printf("  beam_%d: in-place  → phys_ids=", i);
        else if (b == -1)
            printf("  beam_%d: new-block → phys_ids=", i);
        else
            printf("  beam_%d: CoW(%d→%d) → phys_ids=", i, a, b);
        print_ids(bid); printf("\n");
    }


    // ─────────────────────────────────────────────────────────────────────
    // SECTION 3: Prefix Caching
    // ─────────────────────────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 3: Prefix Caching\n");
    printf("============================================================\n\n");

    BlockAllocator gpu3("gpu", 40, BSIZE);

    // 16-token system prompt → 4 full blocks
    std::vector<int> sys_prompt(16);
    std::iota(sys_prompt.begin(), sys_prompt.end(), 100);

    int total_hits = 0, total_blocks_used = 0;
    for (int req = 0; req < 5; ++req) {
        std::vector<int> user_tokens = {200 + req*10, 201 + req*10,
                                        202 + req*10, 203 + req*10};
        std::vector<int> all_tokens = sys_prompt;
        all_tokens.insert(all_tokens.end(), user_tokens.begin(), user_tokens.end());

        BlockTable bt(BSIZE, &gpu3);
        int hits = bt.allocate(all_tokens);
        total_hits        += hits;
        total_blocks_used += (int)bt.physical_block_ids().size();

        printf("  req_%d: %d tokens, phys=", req, (int)all_tokens.size());
        print_ids(bt.physical_block_ids());
        printf(", prefix_hits=%d/4\n", hits);
    }
    printf("  Overall hit rate: %d/%d blocks = %.1f%%\n",
           total_hits, total_blocks_used,
           100.0f * total_hits / total_blocks_used);


    // ─────────────────────────────────────────────────────────────────────
    // SECTION 4: GPU ↔ CPU Swap (simulated)
    // ─────────────────────────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 4: GPU ↔ CPU Swap (simulated)\n");
    printf("============================================================\n\n");

    BlockAllocator gpu4("gpu", 8, BSIZE);
    BlockAllocator cpu4("cpu", 16, BSIZE);

    BlockTable bt_a4(BSIZE, &gpu4), bt_b4(BSIZE, &gpu4);
    std::vector<int> tok_a(8), tok_b(8);
    std::iota(tok_a.begin(), tok_a.end(), 0);
    std::iota(tok_b.begin(), tok_b.end(), 8);
    bt_a4.allocate(tok_a); bt_b4.allocate(tok_b);

    printf("After two prefills: GPU free=%d\n", gpu4.num_free_blocks());
    auto b4_ids = bt_b4.physical_block_ids();

    printf("Swap out seq_b (GPU → CPU):\n");
    swap_out(bt_b4, gpu4, cpu4);
    printf("  GPU free=%d\n", gpu4.num_free_blocks());

    printf("Admit new seq_c (now fits):\n");
    BlockTable bt_c4(BSIZE, &gpu4);
    std::vector<int> tok_c(8); std::iota(tok_c.begin(), tok_c.end(), 16);
    bt_c4.allocate(tok_c);
    printf("  seq_c phys_ids="); print_ids(bt_c4.physical_block_ids()); printf("\n");

    printf("seq_c done — swap seq_b back:\n");
    bt_c4.free_all();
    // For demo, use original CPU ids [0..1] (simulation)
    swap_in(bt_b4, gpu4, cpu4, {0, 1});
    printf("  GPU free=%d\n", gpu4.num_free_blocks());


    // ─────────────────────────────────────────────────────────────────────
    // SECTION 5: Paged Attention Kernel Correctness
    // ─────────────────────────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 5: Paged vs Standard Attention\n");
    printf("============================================================\n\n");

    const int D_K5 = 8, BSIZE5 = 4, N_PHYS5 = 6, N_TOK5 = 10;
    std::mt19937 rng5(42);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    // Build KV store: [N_PHYS5][block_size5][2][D_K5]
    // We use [phys][slot][k_or_v] for indexing
    KVStore kv(N_PHYS5, std::vector<std::array<std::vector<float>, 2>>(
        BSIZE5, {std::vector<float>(D_K5), std::vector<float>(D_K5)}));
    for (auto& blk : kv)
        for (auto& slot : blk)
            for (auto& vec : slot)
                for (auto& v : vec) v = norm(rng5);

    // Query
    std::vector<float> q5(D_K5);
    for (auto& v : q5) v = norm(rng5);

    // Scatter: 10 tokens across blocks phys_2(0-3), phys_5(4-7), phys_0(8-9)
    std::vector<int> btable5 = {2, 5, 0};

    // Reference: assemble contiguous K, V
    std::vector<std::vector<float>> K_ref, V_ref;
    // phys_2 slots 0-3, phys_5 slots 0-3, phys_0 slots 0-1
    for (int t = 0; t < 4; ++t) { K_ref.push_back(kv[2][t][0]); V_ref.push_back(kv[2][t][1]); }
    for (int t = 0; t < 4; ++t) { K_ref.push_back(kv[5][t][0]); V_ref.push_back(kv[5][t][1]); }
    for (int t = 0; t < 2; ++t) { K_ref.push_back(kv[0][t][0]); V_ref.push_back(kv[0][t][1]); }

    auto o_ref   = standard_attention(q5, K_ref, V_ref);
    auto o_paged = paged_attention_single_query(q5, kv, btable5, N_TOK5, BSIZE5);

    float max_err = 0;
    for (int i = 0; i < D_K5; ++i)
        max_err = std::max(max_err, std::abs(o_ref[i] - o_paged[i]));

    printf("  Max absolute error (paged vs standard): %.2e\n", max_err);
    assert(max_err < 1e-4f);
    printf("  ✓ Paged attention matches standard attention.\n");

    printf("\nDone.\n");
    return 0;
}

```

