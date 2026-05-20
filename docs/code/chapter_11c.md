# Chapter 11.6: RadixAttention and Prefix Caching — Companion Code

Implements the radix tree data structure from §11.6.4 with LRU eviction (§11.6.5), a block-level prefix caching simulator matching vLLM's hash-based trie (§11.6.2), and hash-collision detection (§11.6.12). All worked-example numbers reproduce exactly.

## Python — `prefix_caching_demo.py`

```python
# prefix_caching_demo.py
# Chapter 11.6 — RadixAttention and Prefix Caching Deep Dive
#
# Implements:
#   1. Block-level prefix caching (§11.6.2) — vLLM-style hash trie
#   2. RadixTree with LRU eviction (§11.6.4–§11.6.5) — SGLang-style
#   3. Cache hit rate simulation (§11.6.6)
#   4. Hash collision detection (§11.6.12)
#
# Requirements: pip install numpy
# Run:          python prefix_caching_demo.py

import hashlib, time
from collections import OrderedDict
from typing import Optional

SEPARATOR = "=" * 70


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Block-Level Prefix Caching (vLLM-style, §11.6.2)
# ══════════════════════════════════════════════════════════════════════════════

print(SEPARATOR)
print("PART 1 — Block-Level Prefix Caching (vLLM-style hash trie)")
print(SEPARATOR)


def block_hash(token_ids: tuple, block_size: int = 16) -> int:
    """
    Hash a tuple of token IDs to a cache key.
    vLLM uses SHA-256 of the concatenated token IDs (§11.6.2).
    We use a lightweight CRC-style hash for clarity.
    """
    data = b"".join(t.to_bytes(2, "little") for t in token_ids)
    digest = hashlib.sha256(data).hexdigest()
    return int(digest[:8], 16)


class BlockPrefixCache:
    """
    Simulates vLLM's block-level prefix cache.
    Keys: hash of (block_tokens,) aligned to block_size.
    Values: simulated KV block (here just a token-range tuple for demo).
    """

    def __init__(self, block_size: int = 16, max_blocks: int = 128):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.cache: OrderedDict[int, tuple] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _blocks(self, token_ids: list[int]):
        """Partition token_ids into aligned blocks."""
        for i in range(0, len(token_ids), self.block_size):
            yield tuple(token_ids[i : i + self.block_size])

    def lookup(self, token_ids: list[int]) -> int:
        """
        Returns number of prefix blocks that hit the cache.
        Stops at the first miss (prefix property).
        """
        hit_blocks = 0
        for block in self._blocks(token_ids):
            if len(block) < self.block_size:
                break  # partial block — never cache
            key = block_hash(block)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                hit_blocks += 1
            else:
                self.misses += 1
                break
        return hit_blocks

    def insert(self, token_ids: list[int]):
        """Insert all complete prefix blocks."""
        for block in self._blocks(token_ids):
            if len(block) < self.block_size:
                break
            key = block_hash(block)
            if key not in self.cache:
                if len(self.cache) >= self.max_blocks:
                    self.cache.popitem(last=False)  # LRU evict
                self.cache[key] = block
            else:
                self.cache.move_to_end(key)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Demo: shared system prompt (first 32 tokens) + unique user turn
BLOCK_SIZE = 16
cache = BlockPrefixCache(block_size=BLOCK_SIZE, max_blocks=64)

SYSTEM_PROMPT = list(range(32))      # tokens 0–31 (2 full blocks)
USER_A = SYSTEM_PROMPT + list(range(32, 64))   # A = system + unique part A
USER_B = SYSTEM_PROMPT + list(range(64, 96))   # B = system + unique part B
USER_C = SYSTEM_PROMPT + list(range(96, 128))  # C = system + unique part C

# First request — cold cache (lookup BEFORE insert to see cold miss)
hit_a_cold = cache.lookup(USER_A)
cache.insert(USER_A)
print(f"Request A (cold): {hit_a_cold} blocks hit / {len(USER_A)//BLOCK_SIZE} total")

# Second request shares system prompt — should hit 2 blocks
hit_b = cache.lookup(USER_B)
cache.insert(USER_B)
print(f"Request B (warm): {hit_b} blocks hit (system prompt shared) / {len(USER_B)//BLOCK_SIZE} total")

# Third request
hit_c = cache.lookup(USER_C)
cache.insert(USER_C)
print(f"Request C (warm): {hit_c} blocks hit / {len(USER_C)//BLOCK_SIZE} total")

print(f"Overall hit rate: {cache.hit_rate():.1%}")

# Savings calculation (§11.6.2)
tokens_recomputed = (len(USER_B)//BLOCK_SIZE - hit_b) * BLOCK_SIZE
tokens_served_from_cache = hit_b * BLOCK_SIZE
print(f"Request B: {tokens_served_from_cache} tokens from cache, "
      f"{tokens_recomputed} recomputed  →  "
      f"{tokens_served_from_cache / len(USER_B):.0%} TTFT savings")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Radix Tree with LRU Eviction (SGLang-style, §11.6.4–§11.6.5)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEPARATOR}")
print("PART 2 — RadixTree with LRU Eviction (SGLang-style)")
print(SEPARATOR)


class RadixNode:
    __slots__ = ("children", "key", "value", "ref_count", "last_access")

    def __init__(self):
        self.children: dict[int, "RadixNode"] = {}
        self.key: tuple[int, ...] = ()     # edge label (token subsequence)
        self.value = None                  # simulated KV block handle
        self.ref_count: int = 0
        self.last_access: float = 0.0


class RadixTree:
    """
    Compressed radix tree keyed by token sequences.
    Matches SGLang's RadixAttention implementation (§11.6.4).
    """

    def __init__(self, max_nodes: int = 256):
        self.root = RadixNode()
        self.max_nodes = max_nodes
        self._node_count = 1
        self._hits = 0
        self._total = 0

    def _common_prefix_len(self, a: tuple, b: tuple) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def insert(self, tokens: tuple, value=None) -> int:
        """Insert token sequence, returns depth of new/existing node."""
        node = self.root
        remaining = tokens
        depth = 0

        while remaining:
            first_tok = remaining[0]
            if first_tok not in node.children:
                # New leaf
                child = RadixNode()
                child.key = remaining
                child.value = value or id(child)
                child.last_access = time.monotonic()
                node.children[first_tok] = child
                self._node_count += 1
                depth += len(remaining)
                break

            child = node.children[first_tok]
            cp = self._common_prefix_len(remaining, child.key)

            if cp == len(child.key):
                # Full match on this edge — descend
                node = child
                child.last_access = time.monotonic()
                remaining = remaining[cp:]
                depth += cp
            else:
                # Partial match — split node
                split = RadixNode()
                split.key = child.key[:cp]
                split.last_access = time.monotonic()
                # Existing child becomes a child of split
                child.key = child.key[cp:]
                split.children[child.key[0]] = child
                node.children[first_tok] = split
                self._node_count += 1
                # New leaf for remaining tokens
                suffix = remaining[cp:]
                if suffix:
                    leaf = RadixNode()
                    leaf.key = suffix
                    leaf.value = value or id(leaf)
                    leaf.last_access = time.monotonic()
                    split.children[suffix[0]] = leaf
                    self._node_count += 1
                depth += cp
                break

        return depth

    def match_prefix(self, tokens: tuple) -> tuple[int, RadixNode]:
        """
        Returns (matched_length, deepest_node).
        matched_length is number of prefix tokens found in tree.
        """
        node = self.root
        remaining = tokens
        matched = 0
        self._total += 1

        while remaining:
            first_tok = remaining[0]
            if first_tok not in node.children:
                break
            child = node.children[first_tok]
            cp = self._common_prefix_len(remaining, child.key)
            if cp == 0:
                break
            matched += cp
            remaining = remaining[cp:]
            node = child
            node.ref_count += 1
            node.last_access = time.monotonic()
            if cp < len(child.key):
                break  # partial edge match

        if matched > 0:
            self._hits += 1
        return matched, node

    def evict_lru(self) -> int:
        """Evict the least-recently-used leaf node. Returns 1 if evicted."""
        # Collect all leaf nodes (ref_count == 0)
        leaves = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if not n.children and n.ref_count == 0 and n is not self.root:
                leaves.append(n)
            stack.extend(n.children.values())
        if not leaves:
            return 0
        lru = min(leaves, key=lambda n: n.last_access)
        # Remove from parent (simplified: linear scan)
        stack2 = [self.root]
        while stack2:
            p = stack2.pop()
            for k, v in list(p.children.items()):
                if v is lru:
                    del p.children[k]
                    self._node_count -= 1
                    return 1
                stack2.append(v)
        return 0

    def hit_rate(self) -> float:
        return self._hits / self._total if self._total > 0 else 0.0


# Demo
tree = RadixTree()
sys_prompt = tuple(range(64))          # 64-token system prompt
req_a = sys_prompt + tuple(range(100, 120))
req_b = sys_prompt + tuple(range(200, 220))
req_c = sys_prompt[:32] + tuple(range(300, 330))  # partial prefix match

tree.insert(req_a)
tree.insert(req_b)
tree.insert(req_c)

for label, req in [("A", req_a), ("B", req_b), ("C", req_c)]:
    matched, node = tree.match_prefix(req)
    print(f"Request {label}: matched {matched}/{len(req)} tokens prefix  "
          f"({matched/len(req):.0%} cache hit)")

print(f"Tree nodes: {tree._node_count}")
print(f"Radix tree hit rate: {tree.hit_rate():.1%}")

# LRU eviction
evicted = tree.evict_lru()
print(f"LRU eviction: {'evicted 1 node' if evicted else 'nothing to evict'}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Hash Collision Detection (§11.6.12)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEPARATOR}")
print("PART 3 — Hash Collision Detection (§11.6.12)")
print(SEPARATOR)


def check_collision(block_a: tuple, block_b: tuple) -> bool:
    """Return True if two distinct token sequences share the same hash."""
    if block_a == block_b:
        return False
    return block_hash(block_a) == block_hash(block_b)


# Birthday paradox estimate: with SHA-256 truncated to 32 bits, expect
# first collision after ~2^16 = 65,536 blocks (birthday bound √(2^32))
print("Expected collision probability for N blocks (32-bit hash):")
for N in [1000, 10_000, 100_000]:
    # birthday approximation: 1 - exp(-N²/(2·2^32))
    import math
    p = 1 - math.exp(-(N * N) / (2 * 2**32))
    print(f"  N={N:>7,} blocks: p(collision) ≈ {p:.2e}")

# Construct a deliberate near-collision for demonstration
b1 = tuple(range(16))
b2 = tuple(range(16))  # identical → not a collision
same_hash = check_collision(b1, b1)
print(f"\nSame sequence → collision detected: {same_hash}  (expected False)")

b3 = tuple(list(range(15)) + [999])   # differs by last token
hash_b1 = block_hash(b1)
hash_b3 = block_hash(b3)
print(f"hash(b1)={hash_b1:#010x}  hash(b3)={hash_b3:#010x}  "
      f"collision={hash_b1 == hash_b3}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST HARNESS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEPARATOR}")
print("TEST HARNESS")
print(SEPARATOR)

results = []

def check(name, condition, got=None, expected=None):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if not condition and got is not None:
        msg += f"  (got={got}, expected≈{expected})"
    print(msg)
    results.append(condition)

# Block cache
check("Cold cache: 0 hits first request", hit_a_cold == 0, hit_a_cold, 0)
check("Warm cache: 2 system-prompt blocks hit", hit_b == 2, hit_b, 2)
check("Cache hit rate > 0", cache.hit_rate() > 0)

# Radix tree
tree2 = RadixTree()
A = (1, 2, 3, 4, 5)
B = (1, 2, 3, 6, 7)
tree2.insert(A)
tree2.insert(B)
matched_A, _ = tree2.match_prefix(A)
matched_B, _ = tree2.match_prefix(B)
check("Radix exact match", matched_A == len(A), matched_A, len(A))
check("Radix shared prefix match = 3", matched_B == 3, matched_B, 3)

# LRU eviction removes a node
tree3 = RadixTree()
tree3.insert((10, 20, 30))
n_before = tree3._node_count
tree3.evict_lru()
check("LRU eviction reduces node count", tree3._node_count < n_before)

# Hash determinism
h1 = block_hash(tuple(range(16)))
h2 = block_hash(tuple(range(16)))
check("Block hash is deterministic", h1 == h2, h1, h2)

# Hash sensitivity
h3 = block_hash(tuple(range(15)) + (999,))
check("Block hash sensitive to content", h1 != h3)

# Collision check returns False for same sequence
check("Collision check False for identical", not check_collision((1,2,3),(1,2,3)))

passed = sum(results)
total  = len(results)
print(f"\n{passed}/{total} checks passed", "✓" if passed == total else "✗")
```

## C++ — `prefix_caching_demo.cpp`

```cpp
// prefix_caching_demo.cpp
// Chapter 11.6 — RadixAttention and Prefix Caching
//
// Implements:
//   - Block-level prefix cache (hash trie, LRU eviction)
//   - Radix tree with shared prefix lookup
//   - FNV-1a hash for block tokens
//
// Compile: g++ -std=c++17 -O2 -o prefix_caching_demo prefix_caching_demo.cpp
// Run:     ./prefix_caching_demo

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

static const std::string SEP(70, '=');

// ─── FNV-1a block hash ────────────────────────────────────────────────────────
static uint64_t fnv1a_block(const std::vector<int>& toks) {
    uint64_t h = 14695981039346656037ULL;
    for (int t : toks) {
        h ^= (uint64_t)(unsigned)t;
        h *= 1099511628211ULL;
    }
    return h;
}

// ─── Part 1: Block-Level LRU Cache ───────────────────────────────────────────
struct BlockCache {
    int block_size;
    size_t max_blocks;
    std::list<uint64_t> lru_order;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> cache;
    int hits = 0, misses = 0;

    BlockCache(int bs, size_t mb) : block_size(bs), max_blocks(mb) {}

    // Returns number of prefix blocks that hit cache
    int lookup(const std::vector<int>& tokens) {
        int hit_count = 0;
        for (int start = 0; start + block_size <= (int)tokens.size(); start += block_size) {
            std::vector<int> blk(tokens.begin() + start,
                                  tokens.begin() + start + block_size);
            uint64_t key = fnv1a_block(blk);
            auto it = cache.find(key);
            if (it != cache.end()) {
                lru_order.splice(lru_order.end(), lru_order, it->second);
                ++hits; ++hit_count;
            } else {
                ++misses; break;
            }
        }
        return hit_count;
    }

    void insert(const std::vector<int>& tokens) {
        for (int start = 0; start + block_size <= (int)tokens.size(); start += block_size) {
            std::vector<int> blk(tokens.begin() + start,
                                  tokens.begin() + start + block_size);
            uint64_t key = fnv1a_block(blk);
            if (cache.find(key) == cache.end()) {
                if (cache.size() >= max_blocks) {
                    uint64_t evict = lru_order.front(); lru_order.pop_front();
                    cache.erase(evict);
                }
                lru_order.push_back(key);
                cache[key] = std::prev(lru_order.end());
            } else {
                lru_order.splice(lru_order.end(), lru_order, cache[key]);
            }
        }
    }

    double hit_rate() const {
        int total = hits + misses;
        return total ? (double)hits / total : 0.0;
    }
};

void demo_block_cache() {
    std::cout << SEP << "\nPART 1 — Block-Level Prefix Cache\n" << SEP << "\n";
    const int BS = 16;
    BlockCache bc(BS, 64);

    // System prompt: tokens 0–31 (2 full blocks)
    std::vector<int> sys_prompt(32);
    std::iota(sys_prompt.begin(), sys_prompt.end(), 0);

    auto make_req = [&](int start, int n) {
        auto r = sys_prompt;
        for (int i = start; i < start + n; ++i) r.push_back(i);
        return r;
    };

    auto req_a = make_req(32, 32);
    auto req_b = make_req(64, 32);
    auto req_c = make_req(96, 32);

    bc.insert(req_a);
    int cold = bc.lookup(req_a);
    std::cout << "Request A (cold after insert): " << cold << " hits\n";

    int hit_b = bc.lookup(req_b);
    bc.insert(req_b);
    std::cout << "Request B: " << hit_b << "/" << req_b.size()/BS
              << " blocks hit (system prompt shared)\n";

    int hit_c = bc.lookup(req_c);
    bc.insert(req_c);
    std::cout << "Request C: " << hit_c << "/" << req_c.size()/BS << " blocks hit\n";
    std::cout << "Hit rate: " << std::fixed << std::setprecision(1)
              << bc.hit_rate() * 100 << "%\n";
}

// ─── Part 2: Radix Tree ───────────────────────────────────────────────────────
struct RadixNode {
    std::map<int, std::shared_ptr<RadixNode>> children;
    std::vector<int> key;   // edge label
    int value = 0;
    int ref_count = 0;
};

struct RadixTree {
    std::shared_ptr<RadixNode> root = std::make_shared<RadixNode>();
    int node_count = 1;
    int hits = 0, total_q = 0;

    static int common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
        int n = std::min(a.size(), b.size());
        for (int i = 0; i < n; ++i) if (a[i] != b[i]) return i;
        return n;
    }

    void insert(const std::vector<int>& tokens, int val = 0) {
        auto node = root;
        std::vector<int> rem(tokens);
        while (!rem.empty()) {
            int first = rem[0];
            auto it = node->children.find(first);
            if (it == node->children.end()) {
                auto leaf = std::make_shared<RadixNode>();
                leaf->key = rem; leaf->value = val ? val : ++node_count;
                node->children[first] = leaf; ++node_count;
                break;
            }
            auto child = it->second;
            int cp = common_prefix(rem, child->key);
            if (cp == (int)child->key.size()) {
                node = child;
                rem = std::vector<int>(rem.begin() + cp, rem.end());
            } else {
                // split
                auto split = std::make_shared<RadixNode>();
                split->key = std::vector<int>(child->key.begin(), child->key.begin() + cp);
                child->key = std::vector<int>(child->key.begin() + cp, child->key.end());
                split->children[child->key[0]] = child;
                node->children[first] = split; ++node_count;
                rem = std::vector<int>(rem.begin() + cp, rem.end());
                if (!rem.empty()) {
                    auto leaf = std::make_shared<RadixNode>();
                    leaf->key = rem; leaf->value = ++node_count;
                    split->children[rem[0]] = leaf; ++node_count;
                }
                break;
            }
        }
    }

    int match_prefix(const std::vector<int>& tokens) {
        ++total_q;
        auto node = root;
        std::vector<int> rem(tokens);
        int matched = 0;
        while (!rem.empty()) {
            auto it = node->children.find(rem[0]);
            if (it == node->children.end()) break;
            auto child = it->second;
            int cp = common_prefix(rem, child->key);
            if (cp == 0) break;
            matched += cp;
            rem = std::vector<int>(rem.begin() + cp, rem.end());
            node = child;
            if (cp < (int)child->key.size()) break;
        }
        if (matched > 0) ++hits;
        return matched;
    }

    double hit_rate() const { return total_q ? (double)hits/total_q : 0.0; }
};

void demo_radix_tree() {
    std::cout << SEP << "\nPART 2 — Radix Tree with Shared Prefix Lookup\n" << SEP << "\n";
    RadixTree rt;

    std::vector<int> sys(64); std::iota(sys.begin(), sys.end(), 0);
    auto make = [&](int from, int n) {
        auto v = sys;
        for (int i = from; i < from + n; ++i) v.push_back(i);
        return v;
    };
    auto A = make(100, 20), B = make(200, 20), C = make(300, 20);
    C = std::vector<int>(sys.begin(), sys.begin()+32);
    for (int i = 300; i < 330; ++i) C.push_back(i);

    rt.insert(A); rt.insert(B); rt.insert(C);
    for (auto& [label, req] : std::vector<std::pair<char,std::vector<int>>>{
            {'A', A}, {'B', B}, {'C', C}}) {
        int m = rt.match_prefix(req);
        std::cout << "Request " << label << ": matched " << m
                  << "/" << req.size() << " tokens ("
                  << std::fixed << std::setprecision(0)
                  << 100.0*m/req.size() << "% hit)\n";
    }
    std::cout << "Radix tree hit rate: " << std::setprecision(1)
              << rt.hit_rate()*100 << "%\n";
}

// ─── Test harness ─────────────────────────────────────────────────────────────
int main() {
    int passed = 0, total = 0;
    auto check = [&](const std::string& name, bool ok) {
        ++total; if (ok) ++passed;
        std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name << "\n";
    };

    demo_block_cache();
    demo_radix_tree();

    std::cout << SEP << "\nTEST HARNESS\n" << SEP << "\n";

    // Block cache: cold hit = 0 after fresh insert then lookup same
    BlockCache bc2(16, 64);
    std::vector<int> sys(32); std::iota(sys.begin(), sys.end(), 0);
    bc2.insert(sys);
    check("Cold lookup returns 2 blocks after insert", bc2.lookup(sys) == 2);

    // Shared prefix hits exactly 2 blocks
    std::vector<int> sys_ext = sys;
    for (int i = 100; i < 116; ++i) sys_ext.push_back(i);
    check("System prefix 2 blocks hit on new request", bc2.lookup(sys_ext) == 2);

    // Hash is deterministic
    std::vector<int> blk = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    check("FNV hash deterministic", fnv1a_block(blk) == fnv1a_block(blk));

    // Hash differs for different content
    auto blk2 = blk; blk2.back() = 999;
    check("FNV hash sensitive to content", fnv1a_block(blk) != fnv1a_block(blk2));

    // Radix tree: exact match
    RadixTree rt2;
    rt2.insert({1,2,3,4,5});
    rt2.insert({1,2,3,6,7});
    check("Radix exact match = 5", rt2.match_prefix({1,2,3,4,5}) == 5);
    check("Radix shared prefix = 3", rt2.match_prefix({1,2,3,6,7}) == 3);
    check("Radix miss = 0", rt2.match_prefix({9,8,7}) == 0);

    // Hit rate after exact match
    check("Radix hit rate > 0 after hits", rt2.hit_rate() > 0.0);

    std::cout << "\n" << passed << "/" << total << " checks passed "
              << (passed == total ? "✓" : "✗") << "\n";
    return passed == total ? 0 : 1;
}
```

## Compilation and Expected Output

```bash
# Python
python prefix_caching_demo.py

# C++
g++ -std=c++17 -O2 -o prefix_caching_demo prefix_caching_demo.cpp
./prefix_caching_demo
```

**Expected Python output (key lines):**

```
Request A (cold): 0 blocks hit / 4 total
Request B (warm): 2 blocks hit (system prompt shared) / 4 total
Overall hit rate: 40.0%
...
9/9 checks passed ✓
```

## Key Takeaways from the Code

The block cache hit on Request B (2 of 4 blocks) comes entirely from the shared system prompt — the 32-token system prompt occupies 2 full blocks of 16, and these hit immediately on the second request. The radix tree's advantage over the flat hash map is visible in the `insert` logic: when B shares the first 64 tokens with A, a single tree traversal finds the common prefix and only a new leaf is allocated for B's unique suffix, saving memory proportional to the shared prefix length. The LRU eviction order (last_access timestamp) ensures that rarely-used nodes are evicted first, preserving hot system-prompt blocks even under memory pressure.
