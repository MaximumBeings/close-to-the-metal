# Chapter 7: The Scheduler and Request Lifecycle — Companion Code

## Python — `scheduler_demo.py`

```python
# scheduler_demo.py
# Chapter 7 — The Scheduler and Request Lifecycle
#
# Standalone companion to chapter_07_scheduler.md.
# Simulates:
#   1. SequenceGroup/Sequence lifecycle (WAITING → RUNNING → FINISHED)
#   2. vLLM-style 7-step scheduling loop
#   3. max_num_seqs and max_num_batched_tokens admission gates
#   4. SWAP and RECOMPUTE preemption policies
#   5. Priority-aware scheduling vs FCFS
#   6. Chunked-prefill simulation
#   7. Throughput statistics as function of batch size
#
# Requirements:
#   pip install numpy  (only for Section 7 plots; otherwise none)
#
# Run:
#   python scheduler_demo.py

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Core data model
# ─────────────────────────────────────────────────────────────────────────────

class SequenceStatus(Enum):
    WAITING  = "waiting"
    RUNNING  = "running"
    SWAPPED  = "swapped"
    FINISHED = "finished"
    ABORTED  = "aborted"


@dataclass
class Sequence:
    """
    Single output sequence within a SequenceGroup.

    In vLLM a SequenceGroup with n=4 has 4 Sequence objects sharing
    the prompt prefix blocks.  Here we simplify: each request has one Sequence.
    """
    seq_id:           int
    prompt_len:       int
    max_new_tokens:   int
    priority:         int   = 0       # higher = more important
    arrival_time:     float = 0.0

    # mutable state
    tokens_generated: int           = field(default=0, init=False)
    status:           SequenceStatus = field(default=SequenceStatus.WAITING,
                                             init=False)
    blocks_allocated: int           = field(default=0, init=False)
    finish_iteration: Optional[int] = field(default=None, init=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.tokens_generated

    def is_finished(self) -> bool:
        return self.tokens_generated >= self.max_new_tokens

    def __lt__(self, other: "Sequence") -> bool:
        """Higher priority first; break ties by arrival time (FIFO)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.arrival_time < other.arrival_time

    def __repr__(self) -> str:
        return (f"Seq(id={self.seq_id}, prompt={self.prompt_len}, "
                f"gen={self.tokens_generated}/{self.max_new_tokens}, "
                f"status={self.status.value})")


# ─────────────────────────────────────────────────────────────────────────────
# Simplified block manager (flat accounting — not PagedAttention)
# ─────────────────────────────────────────────────────────────────────────────

class BlockManager:
    """
    Simulates block allocation without the full PagedAttention logic.

    We track how many blocks each sequence has allocated and whether
    the free pool can satisfy new requests.

    This is enough to drive the scheduling simulation; for the full
    PagedAttention block manager see chapter_06/paged_attention_demo.py.
    """

    def __init__(self, total_blocks: int, block_size: int) -> None:
        self.total_blocks = total_blocks
        self.block_size   = block_size
        self._used:  Dict[int, int] = {}   # seq_id → blocks

    # ── allocation helpers ────────────────────────────────────────────────

    def blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence) -> bool:
        needed = self.blocks_for(seq.prompt_len)
        return needed <= self.free_blocks

    def allocate(self, seq: Sequence) -> None:
        n = self.blocks_for(seq.prompt_len)
        self._used[seq.seq_id] = n
        seq.blocks_allocated   = n

    def can_append_slot(self, seq: Sequence) -> bool:
        """True if the next token fits in current blocks or a free block exists."""
        if seq.seq_id not in self._used:
            return False   # sequence not yet allocated (should not happen, but guard)
        current_slots = self._used[seq.seq_id] * self.block_size
        if seq.total_tokens + 1 <= current_slots:
            return True
        return self.free_blocks >= 1

    def append_slot(self, seq: Sequence) -> bool:
        current_slots = self._used[seq.seq_id] * self.block_size
        if seq.total_tokens + 1 <= current_slots:
            return True
        if self.free_blocks >= 1:
            self._used[seq.seq_id] += 1
            seq.blocks_allocated   += 1
            return True
        return False

    def free(self, seq: Sequence) -> None:
        n = self._used.pop(seq.seq_id, 0)
        seq.blocks_allocated = 0

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - sum(self._used.values())

    @property
    def used_blocks(self) -> int:
        return sum(self._used.values())

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.total_blocks if self.total_blocks else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler config
# ─────────────────────────────────────────────────────────────────────────────

class PreemptionPolicy(Enum):
    SWAP      = "swap"
    RECOMPUTE = "recompute"


@dataclass
class SchedulerConfig:
    max_num_seqs:            int              = 64
    max_num_batched_tokens:  int              = 2048
    preemption_policy:       PreemptionPolicy = PreemptionPolicy.RECOMPUTE
    enable_priority:         bool             = False
    chunked_prefill:         bool             = False
    chunk_size:              int              = 512


# ─────────────────────────────────────────────────────────────────────────────
# Iteration stats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IterStats:
    iteration:     int
    n_prefill:     int
    n_decode:      int
    total_tokens:  int
    n_preemptions: int
    n_waiting:     int
    n_running:     int
    n_swapped:     int
    free_blocks:   int
    utilization:   float


# ─────────────────────────────────────────────────────────────────────────────
# The Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class Scheduler:
    """
    Simulates vLLM's scheduling loop.

    The 7 steps (see §7.2):
      1. Evict finished sequences
      2. Try swap-ins (SWAPPED → RUNNING)
      3. Check append-slot for all running sequences
      4. Preempt if memory pressure (RUNNING → SWAPPED or WAITING)
      5. Admit new requests (WAITING → RUNNING)
      6. Build batch (prefill + decode groups)
      7. Simulate one decode/prefill step
    """

    def __init__(self, cfg: SchedulerConfig, bm: BlockManager) -> None:
        self.cfg     = cfg
        self.bm      = bm
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence]  = []
        self.swapped: List[Sequence]  = []
        self._iter   = 0
        self._total_tokens_generated = 0
        self._total_preemptions      = 0

    def add_request(self, seq: Sequence) -> None:
        seq.arrival_time = float(self._iter)
        seq.status       = SequenceStatus.WAITING
        self.waiting.append(seq)

    # ── Main scheduling step ──────────────────────────────────────────────

    def step(self) -> IterStats:
        self._iter += 1
        n_preemptions = 0

        # ── Step 1: Evict finished ────────────────────────────────────────
        still_running = []
        for seq in self.running:
            if seq.is_finished():
                seq.status           = SequenceStatus.FINISHED
                seq.finish_iteration = self._iter
                self.bm.free(seq)
            else:
                still_running.append(seq)
        self.running = still_running

        # ── Step 2: Swap-ins ──────────────────────────────────────────────
        for seq in list(self.swapped):
            if self.bm.can_allocate(seq):
                self.bm.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                self.swapped.remove(seq)

        # ── Steps 3+4: Ensure append slots; preempt if necessary ─────────
        for seq in list(self.running):
            if seq not in self.running:   # may have been evicted as victim
                continue
            while not self.bm.can_append_slot(seq):
                if not self.running:
                    break
                # Select victim
                if self.cfg.enable_priority:
                    victim = min(self.running, key=lambda s: (s.priority, -s.arrival_time))
                else:
                    victim = self.running[-1]   # LIFO

                self.running.remove(victim)
                self.bm.free(victim)
                n_preemptions += 1

                if self.cfg.preemption_policy == PreemptionPolicy.SWAP:
                    victim.status = SequenceStatus.SWAPPED
                    self.swapped.append(victim)
                else:
                    # RECOMPUTE: reset and re-enqueue
                    victim.tokens_generated = 0
                    victim.status           = SequenceStatus.WAITING
                    self.waiting.appendleft(victim)

        # ── Step 5: Admit new requests ────────────────────────────────────
        token_budget = self.cfg.max_num_batched_tokens - len(self.running)
        seq_budget   = self.cfg.max_num_seqs           - len(self.running)

        if self.cfg.enable_priority:
            candidates = sorted(self.waiting)   # __lt__ → priority descending
        else:
            candidates = list(self.waiting)     # FIFO order

        for seq in candidates:
            if seq not in self.waiting:
                continue
            if seq_budget <= 0 or token_budget <= 0:
                break

            # How many tokens to admit in chunked prefill
            admit_tokens = (min(seq.prompt_len, self.cfg.chunk_size)
                            if self.cfg.chunked_prefill
                            else seq.prompt_len)

            if admit_tokens > token_budget:
                continue
            if not self.bm.can_allocate(seq):
                break

            self.bm.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            self.waiting.remove(seq)
            token_budget -= admit_tokens
            seq_budget   -= 1

        # ── Steps 6+7: Execute one step for each running sequence ─────────
        n_prefill = 0
        n_decode  = 0
        total_tok = 0

        for seq in list(self.running):
            if seq.tokens_generated == 0:
                # Prefill
                n_prefill += 1
                total_tok += (min(seq.prompt_len, self.cfg.chunk_size)
                              if self.cfg.chunked_prefill else seq.prompt_len)
            else:
                # Decode: one token
                n_decode  += 1
                total_tok += 1

            if self.bm.append_slot(seq):
                seq.tokens_generated += 1
                self._total_tokens_generated += 1

        self._total_preemptions += n_preemptions

        return IterStats(
            iteration     = self._iter,
            n_prefill     = n_prefill,
            n_decode      = n_decode,
            total_tokens  = total_tok,
            n_preemptions = n_preemptions,
            n_waiting     = len(self.waiting),
            n_running     = len(self.running),
            n_swapped     = len(self.swapped),
            free_blocks   = self.bm.free_blocks,
            utilization   = self.bm.utilization,
        )

    @property
    def is_idle(self) -> bool:
        return not (self.waiting or self.running or self.swapped)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_requests(n: int, seed: int = 42,
                  prompt_range=(32, 256),
                  gen_range=(20, 100),
                  priority_range=(0, 0)) -> List[Sequence]:
    rng = random.Random(seed)
    return [
        Sequence(
            seq_id         = i,
            prompt_len     = rng.randint(*prompt_range),
            max_new_tokens = rng.randint(*gen_range),
            priority       = rng.randint(*priority_range),
        )
        for i in range(n)
    ]


def run_until_done(sched: Scheduler,
                   max_iters: int = 2000,
                   verbose: bool  = False) -> Tuple[int, int]:
    """Returns (total_iterations, total_tokens)."""
    for i in range(max_iters):
        s = sched.step()
        if verbose:
            print(f"  iter={s.iteration:4d}  pf={s.n_prefill}  dc={s.n_decode}  "
                  f"tok={s.total_tokens:5d}  wait={s.n_waiting}  "
                  f"run={s.n_running}  free_blk={s.free_blocks}  "
                  f"preempt={s.n_preemptions}")
        if sched.is_idle:
            return s.iteration, sched._total_tokens_generated
    return max_iters, sched._total_tokens_generated


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Basic lifecycle trace
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("SECTION 1: Basic Lifecycle Trace (8 requests)")
print("=" * 65)

reqs1 = make_requests(8, seed=42, prompt_range=(20, 150), gen_range=(10, 40))
for r in reqs1:
    print(f"  seq_{r.seq_id}: prompt={r.prompt_len}, max_new={r.max_new_tokens}")

cfg1  = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=512)
bm1   = BlockManager(total_blocks=200, block_size=16)
sched1 = Scheduler(cfg1, bm1)
for r in reqs1:
    sched1.add_request(r)

print(f"\n{'Iter':<5} {'pf':<4} {'dc':<4} {'tok':<6} "
      f"{'wait':<5} {'run':<4} {'swap':<4} {'free_blk':<9} {'preempt'}")
print("-" * 55)

for _ in range(80):
    s = sched1.step()
    print(f"  {s.iteration:<4} {s.n_prefill:<4} {s.n_decode:<4} {s.total_tokens:<6} "
          f"{s.n_waiting:<5} {s.n_running:<4} {s.n_swapped:<4} {s.free_blocks:<9} {s.n_preemptions}")
    if sched1.is_idle:
        print(f"\n  ✓ All requests finished at iteration {s.iteration}.")
        break


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Preemption under memory pressure
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SECTION 2: Preemption Under Memory Pressure (tight pool)")
print("=" * 65)

for policy in [PreemptionPolicy.SWAP, PreemptionPolicy.RECOMPUTE]:
    reqs2 = make_requests(12, seed=7, prompt_range=(80, 120), gen_range=(25, 35))
    cfg2  = SchedulerConfig(max_num_seqs=8,
                            max_num_batched_tokens=2048,
                            preemption_policy=policy)
    bm2   = BlockManager(total_blocks=60, block_size=16)   # intentionally tight
    sched2 = Scheduler(cfg2, bm2)
    for r in reqs2:
        sched2.add_request(r)

    iters, total_tok = run_until_done(sched2)
    print(f"\n  Policy={policy.value:10s}: "
          f"iters={iters}, "
          f"total_tokens={total_tok}, "
          f"preemptions={sched2._total_preemptions}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Priority scheduling — HP vs LP finish times
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SECTION 3: Priority-Aware vs FCFS")
print("=" * 65)

def priority_experiment(enable_priority: bool) -> Tuple[float, float]:
    """Returns (hp_avg_finish, lp_avg_finish)."""
    hp_reqs = [Sequence(seq_id=i,   prompt_len=50,  max_new_tokens=20, priority=10)
               for i in range(4)]
    lp_reqs = [Sequence(seq_id=i+4, prompt_len=200, max_new_tokens=80, priority=1)
               for i in range(4)]

    cfg = SchedulerConfig(max_num_seqs=4,
                          max_num_batched_tokens=1024,
                          enable_priority=enable_priority)
    bm  = BlockManager(total_blocks=500, block_size=16)
    sch = Scheduler(cfg, bm)
    for r in hp_reqs + lp_reqs:
        sch.add_request(r)

    for _ in range(1000):
        sch.step()
        if sch.is_idle:
            break

    hp_finish = [r.finish_iteration for r in hp_reqs if r.finish_iteration]
    lp_finish = [r.finish_iteration for r in lp_reqs if r.finish_iteration]
    return (sum(hp_finish) / len(hp_finish) if hp_finish else 999,
            sum(lp_finish) / len(lp_finish) if lp_finish else 999)

hp_fcfs, lp_fcfs = priority_experiment(False)
hp_prio, lp_prio = priority_experiment(True)

print(f"\n  {'Policy':<22} {'HP avg finish':<16} {'LP avg finish'}")
print("  " + "-" * 50)
print(f"  {'FCFS':<22} {hp_fcfs:<16.1f} {lp_fcfs:.1f}")
print(f"  {'Priority-aware':<22} {hp_prio:<16.1f} {lp_prio:.1f}")
print(f"\n  HP speedup from priority: {hp_fcfs/hp_prio:.2f}×")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Chunked prefill vs standard
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SECTION 4: Chunked Prefill — Decode Latency Impact")
print("=" * 65)

def chunked_experiment(chunked: bool, chunk_sz: int = 256) -> Tuple[float, int]:
    """
    Returns (average decode gap, total iterations).
    Decode gap = iterations between first decode token and finish of decode batch.
    """
    # One very long prefill + 8 decode sequences already running
    long_req   = Sequence(seq_id=0, prompt_len=2048, max_new_tokens=50)
    decode_reqs= [Sequence(seq_id=i+1, prompt_len=32, max_new_tokens=60)
                  for i in range(8)]

    cfg = SchedulerConfig(max_num_seqs=16,
                          max_num_batched_tokens=2048,
                          chunked_prefill=chunked,
                          chunk_size=chunk_sz)
    bm  = BlockManager(total_blocks=1000, block_size=16)
    sch = Scheduler(cfg, bm)

    # Pre-admit decode sequences
    for r in decode_reqs:
        sch.add_request(r)
    for _ in range(5):   # run them for a few steps so they are in RUNNING
        sch.step()

    # Now add the long prefill
    sch.add_request(long_req)

    # Count iterations where decode seqs are starved (n_decode < 8)
    starvation_iters = 0
    total_iters      = 0
    for _ in range(200):
        s = sch.step()
        total_iters += 1
        if s.n_decode < 8 and not sch.is_idle:
            starvation_iters += 1
        if sch.is_idle:
            break

    return starvation_iters, total_iters

starve_std,  iters_std  = chunked_experiment(False)
starve_chk,  iters_chk  = chunked_experiment(True, chunk_sz=256)

print(f"\n  {'Mode':<20} {'Decode-starved iters':<24} {'Total iters'}")
print("  " + "-" * 55)
print(f"  {'Standard prefill':<20} {starve_std:<24} {iters_std}")
print(f"  {'Chunked (size=256)':<20} {starve_chk:<24} {iters_chk}")
print(f"\n  Chunked prefill reduced decode starvation by "
      f"{(starve_std - starve_chk) / max(starve_std, 1) * 100:.0f}%")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Throughput vs max_num_seqs
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SECTION 5: Throughput (tokens/iter) vs max_num_seqs")
print("=" * 65)

print(f"\n  {'max_num_seqs':<16} {'tokens/iter':<14} {'utilization':<14} {'preemptions'}")
print("  " + "-" * 55)

for max_seqs in [4, 8, 16, 32, 64, 128]:
    random.seed(3)
    cfg = SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=4096)
    bm  = BlockManager(total_blocks=3000, block_size=16)
    sch = Scheduler(cfg, bm)

    seqs_added = 0
    tok_history = []
    util_history = []

    for it in range(200):
        # Continuous arrivals: 4 new requests every 8 iterations
        if it % 8 == 0:
            for _ in range(4):
                r = Sequence(
                    seq_id         = seqs_added,
                    prompt_len     = random.randint(32, 200),
                    max_new_tokens = random.randint(30, 80),
                )
                sch.add_request(r)
                seqs_added += 1

        s = sch.step()
        if it >= 50:   # steady-state only
            tok_history.append(s.total_tokens)
            util_history.append(s.utilization)

    avg_tok  = sum(tok_history)  / max(len(tok_history), 1)
    avg_util = sum(util_history) / max(len(util_history), 1)
    print(f"  {max_seqs:<16} {avg_tok:<14.1f} {avg_util*100:<13.1f}% "
          f"{sch._total_preemptions}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Admission gate — max_num_batched_tokens impact on TTFT
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SECTION 6: TTFT vs max_num_batched_tokens")
print("=" * 65)

print(f"\n  {'max_batched_tok':<18} {'avg TTFT (iters)':<20} {'avg throughput'}")
print("  " + "-" * 55)

for max_tok in [256, 512, 1024, 2048, 4096]:
    random.seed(5)
    cfg = SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=max_tok)
    bm  = BlockManager(total_blocks=2000, block_size=16)
    sch = Scheduler(cfg, bm)

    all_reqs = make_requests(40, seed=5, prompt_range=(100, 400), gen_range=(20, 60))
    for r in all_reqs:
        sch.add_request(r)

    ttft_list    = []
    tok_per_iter = []

    for it in range(500):
        s = sch.step()
        tok_per_iter.append(s.total_tokens)
        for req in all_reqs:
            if (req.finish_iteration is None
                    and req.status == SequenceStatus.RUNNING
                    and req.tokens_generated == 1
                    and req.seq_id not in [x.seq_id for x in []]):
                # First token generated this step
                pass
        # Approximate TTFT as iterations from arrival to first gen
        # (iter - arrival_time for newly-started seqs)
        for req in all_reqs:
            if (req.tokens_generated == 1
                    and not hasattr(req, '_ttft_recorded')):
                req._ttft_recorded = True  # type: ignore[attr-defined]
                ttft_list.append(s.iteration - int(req.arrival_time))

        if sch.is_idle:
            break

    avg_ttft = sum(ttft_list) / max(len(ttft_list), 1)
    avg_tput = sum(tok_per_iter) / max(len(tok_per_iter), 1)
    print(f"  {max_tok:<18} {avg_ttft:<20.1f} {avg_tput:.1f}")

print("\nDone.")

```

## C++ — `scheduler_demo.cpp`

```cpp
// scheduler_demo.cpp
// Chapter 7 — The Scheduler and Request Lifecycle
//
// Implements from scratch (no third-party libraries):
//   1. Sequence / SequenceStatus lifecycle
//   2. Simplified block manager (flat accounting)
//   3. Scheduler: 7-step loop, admission gates, SWAP/RECOMPUTE preemption
//   4. Priority-aware scheduling
//   5. Throughput statistics
//   6. Multi-request llama.cpp-style dispatch (simulated)
//
// Build:
//   g++ -std=c++17 -O2 scheduler_demo.cpp -o scheduler_demo
//
// Run:
//   ./scheduler_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// SequenceStatus
// ─────────────────────────────────────────────────────────────────────────────

enum class SequenceStatus {
    WAITING,
    RUNNING,
    SWAPPED,
    FINISHED,
    ABORTED,
};

static const char* status_str(SequenceStatus s) {
    switch (s) {
        case SequenceStatus::WAITING:  return "waiting";
        case SequenceStatus::RUNNING:  return "running";
        case SequenceStatus::SWAPPED:  return "swapped";
        case SequenceStatus::FINISHED: return "finished";
        default:                       return "aborted";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence
// ─────────────────────────────────────────────────────────────────────────────

struct Sequence {
    int    seq_id;
    int    prompt_len;
    int    max_new_tokens;
    int    priority         = 0;
    float  arrival_time     = 0.0f;

    // mutable
    int    tokens_generated = 0;
    int    blocks_allocated = 0;
    int    finish_iteration = -1;
    SequenceStatus status   = SequenceStatus::WAITING;

    int total_tokens() const { return prompt_len + tokens_generated; }
    bool is_finished()  const { return tokens_generated >= max_new_tokens; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Block manager (flat accounting)
// ─────────────────────────────────────────────────────────────────────────────

class BlockManager {
public:
    BlockManager(int total_blocks, int block_size)
        : total_(total_blocks), bsize_(block_size), used_(0) {}

    int blocks_for(int tokens) const {
        return (tokens + bsize_ - 1) / bsize_;
    }

    bool can_allocate(const Sequence& seq) const {
        return blocks_for(seq.prompt_len) <= free_blocks();
    }

    void allocate(Sequence& seq) {
        int n = blocks_for(seq.prompt_len);
        seq_blocks_[seq.seq_id] = n;
        used_ += n;
        seq.blocks_allocated = n;
    }

    bool can_append(const Sequence& seq) const {
        int slots = seq_blocks_.count(seq.seq_id)
                    ? seq_blocks_.at(seq.seq_id) * bsize_ : 0;
        if (seq.total_tokens() + 1 <= slots) return true;
        return free_blocks() >= 1;
    }

    bool append(Sequence& seq) {
        int slots = seq_blocks_[seq.seq_id] * bsize_;
        if (seq.total_tokens() + 1 <= slots) return true;
        if (free_blocks() >= 1) {
            seq_blocks_[seq.seq_id]++;
            used_++;
            seq.blocks_allocated++;
            return true;
        }
        return false;
    }

    void free(Sequence& seq) {
        auto it = seq_blocks_.find(seq.seq_id);
        if (it != seq_blocks_.end()) {
            used_ -= it->second;
            seq_blocks_.erase(it);
        }
        seq.blocks_allocated = 0;
    }

    int free_blocks()  const { return total_ - used_; }
    int used_blocks()  const { return used_; }
    double utilization() const { return total_ ? (double)used_ / total_ : 0.0; }

private:
    int total_, bsize_, used_;
    std::unordered_map<int, int> seq_blocks_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Scheduler config
// ─────────────────────────────────────────────────────────────────────────────

enum class PreemptionPolicy { SWAP, RECOMPUTE };

struct SchedulerConfig {
    int              max_num_seqs           = 64;
    int              max_num_batched_tokens = 2048;
    PreemptionPolicy preemption_policy      = PreemptionPolicy::RECOMPUTE;
    bool             enable_priority        = false;
    bool             chunked_prefill        = false;
    int              chunk_size             = 512;
};

// ─────────────────────────────────────────────────────────────────────────────
// Iteration stats
// ─────────────────────────────────────────────────────────────────────────────

struct IterStats {
    int    iteration;
    int    n_prefill;
    int    n_decode;
    int    total_tokens;
    int    n_preemptions;
    int    n_waiting;
    int    n_running;
    int    n_swapped;
    int    free_blocks;
    double utilization;
};

// ─────────────────────────────────────────────────────────────────────────────
// Scheduler
// ─────────────────────────────────────────────────────────────────────────────

class Scheduler {
public:
    Scheduler(const SchedulerConfig& cfg, BlockManager& bm)
        : cfg_(cfg), bm_(bm), iter_(0),
          total_tokens_gen_(0), total_preemptions_(0) {}

    void add_request(Sequence seq) {
        seq.arrival_time = (float)iter_;
        seq.status       = SequenceStatus::WAITING;
        waiting_.push_back(std::move(seq));
    }

    IterStats step() {
        ++iter_;
        int n_preemptions = 0;

        // ── Step 1: Evict finished ────────────────────────────────────────
        {
            std::vector<Sequence> still;
            for (auto& seq : running_) {
                if (seq.is_finished()) {
                    seq.status           = SequenceStatus::FINISHED;
                    seq.finish_iteration = iter_;
                    bm_.free(seq);
                    finished_.push_back(seq);
                } else {
                    still.push_back(seq);
                }
            }
            running_ = std::move(still);
        }

        // ── Step 2: Swap-ins ──────────────────────────────────────────────
        for (auto it = swapped_.begin(); it != swapped_.end(); ) {
            if (bm_.can_allocate(*it)) {
                bm_.allocate(*it);
                it->status = SequenceStatus::RUNNING;
                running_.push_back(*it);
                it = swapped_.erase(it);
            } else {
                ++it;
            }
        }

        // ── Steps 3+4: Append-slot check + preemption ─────────────────────
        for (auto& seq : running_) {
            while (!bm_.can_append(seq)) {
                if (running_.empty()) break;

                // Select victim
                auto victim_it = running_.end() - 1;  // LIFO default
                if (cfg_.enable_priority) {
                    victim_it = std::min_element(
                        running_.begin(), running_.end(),
                        [](const Sequence& a, const Sequence& b) {
                            return a.priority != b.priority
                                   ? a.priority < b.priority
                                   : a.arrival_time > b.arrival_time;
                        });
                }

                Sequence victim = *victim_it;
                running_.erase(victim_it);
                bm_.free(victim);
                ++n_preemptions;

                if (cfg_.preemption_policy == PreemptionPolicy::SWAP) {
                    victim.status = SequenceStatus::SWAPPED;
                    swapped_.push_back(victim);
                } else {
                    // RECOMPUTE: reset + re-enqueue
                    victim.tokens_generated = 0;
                    victim.status           = SequenceStatus::WAITING;
                    waiting_.push_front(victim);
                }
            }
        }

        // ── Step 5: Admit new requests ────────────────────────────────────
        int tok_budget = cfg_.max_num_batched_tokens - (int)running_.size();
        int seq_budget = cfg_.max_num_seqs           - (int)running_.size();

        // Sort by priority if enabled
        if (cfg_.enable_priority) {
            std::stable_sort(waiting_.begin(), waiting_.end(),
                             [](const Sequence& a, const Sequence& b) {
                                 return a.priority != b.priority
                                        ? a.priority > b.priority
                                        : a.arrival_time < b.arrival_time;
                             });
        }

        for (auto it = waiting_.begin();
             it != waiting_.end() && seq_budget > 0 && tok_budget > 0; ) {

            int admit_tok = cfg_.chunked_prefill
                            ? std::min(it->prompt_len, cfg_.chunk_size)
                            : it->prompt_len;

            if (admit_tok > tok_budget) { ++it; continue; }
            if (!bm_.can_allocate(*it)) break;

            bm_.allocate(*it);
            it->status = SequenceStatus::RUNNING;
            running_.push_back(*it);
            it = waiting_.erase(it);
            tok_budget -= admit_tok;
            --seq_budget;
        }

        // ── Steps 6+7: Execute one step ───────────────────────────────────
        int n_prefill = 0, n_decode = 0, total_tok = 0;
        for (auto& seq : running_) {
            if (seq.tokens_generated == 0) {
                ++n_prefill;
                total_tok += cfg_.chunked_prefill
                             ? std::min(seq.prompt_len, cfg_.chunk_size)
                             : seq.prompt_len;
            } else {
                ++n_decode;
                ++total_tok;
            }
            if (bm_.append(seq)) {
                ++seq.tokens_generated;
                ++total_tokens_gen_;
            }
        }
        total_preemptions_ += n_preemptions;

        return IterStats{
            iter_, n_prefill, n_decode, total_tok, n_preemptions,
            (int)waiting_.size(), (int)running_.size(), (int)swapped_.size(),
            bm_.free_blocks(), bm_.utilization()
        };
    }

    bool is_idle() const {
        return waiting_.empty() && running_.empty() && swapped_.empty();
    }

    int total_tokens_generated() const { return total_tokens_gen_; }
    int total_preemptions()      const { return total_preemptions_; }
    int iter()                   const { return iter_; }

    const std::vector<Sequence>& finished_seqs() const { return finished_; }

private:
    SchedulerConfig       cfg_;
    BlockManager&         bm_;
    int                   iter_;
    int                   total_tokens_gen_;
    int                   total_preemptions_;

    std::deque<Sequence>  waiting_;
    std::vector<Sequence> running_;
    std::vector<Sequence> swapped_;
    std::vector<Sequence> finished_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<Sequence> make_requests(
    int n, int seed,
    int pmin, int pmax,
    int gmin, int gmax,
    int priority = 0)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pdist(pmin, pmax);
    std::uniform_int_distribution<int> gdist(gmin, gmax);

    std::vector<Sequence> seqs;
    for (int i = 0; i < n; ++i) {
        Sequence s;
        s.seq_id         = i;
        s.prompt_len     = pdist(rng);
        s.max_new_tokens = gdist(rng);
        s.priority       = priority;
        seqs.push_back(s);
    }
    return seqs;
}

static std::pair<int,int> run_until_done(Scheduler& sched, int max_iters = 2000) {
    for (int i = 0; i < max_iters; ++i) {
        sched.step();
        if (sched.is_idle()) return {sched.iter(), sched.total_tokens_generated()};
    }
    return {max_iters, sched.total_tokens_generated()};
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {

    // ─────────────────────────────────────────────────────────────────────
    // SECTION 1: Basic lifecycle trace
    // ─────────────────────────────────────────────────────────────────────
    printf("============================================================\n");
    printf("SECTION 1: Basic Lifecycle Trace (8 requests)\n");
    printf("============================================================\n\n");

    auto reqs1 = make_requests(8, 42, 20, 150, 10, 40);
    for (auto& r : reqs1)
        printf("  seq_%d: prompt=%d, max_new=%d\n", r.seq_id, r.prompt_len, r.max_new_tokens);

    SchedulerConfig cfg1;
    cfg1.max_num_seqs           = 4;
    cfg1.max_num_batched_tokens = 512;

    BlockManager bm1(200, 16);
    Scheduler    sched1(cfg1, bm1);
    for (auto& r : reqs1) sched1.add_request(r);

    printf("\n%-5s %-4s %-4s %-6s %-5s %-4s %-4s %-9s %s\n",
           "Iter", "pf", "dc", "tok", "wait", "run", "swap", "free_blk", "preempt");
    printf("%s\n", std::string(55, '-').c_str());

    for (int i = 0; i < 80; ++i) {
        IterStats s = sched1.step();
        printf("  %-4d %-4d %-4d %-6d %-5d %-4d %-4d %-9d %d\n",
               s.iteration, s.n_prefill, s.n_decode, s.total_tokens,
               s.n_waiting, s.n_running, s.n_swapped, s.free_blocks, s.n_preemptions);
        if (sched1.is_idle()) {
            printf("\n  ✓ All requests finished at iteration %d.\n", s.iteration);
            break;
        }
    }


    // ─────────────────────────────────────────────────────────────────────
    // SECTION 2: Preemption under memory pressure
    // ─────────────────────────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 2: Preemption Under Memory Pressure\n");
    printf("============================================================\n\n");

    for (auto policy : {PreemptionPolicy::SWAP, PreemptionPolicy::RECOMPUTE}) {
        auto reqs2 = make_requests(12, 7, 80, 120, 25, 35);

        SchedulerConfig cfg2;
        cfg2.max_num_seqs           = 8;
        cfg2.max_num_batched_tokens = 2048;
        cfg2.preemption_policy      = policy;

        BlockManager bm2(60, 16);  // tight
        Scheduler    sched2(cfg2, bm2);
        for (auto& r : reqs2) sched2.add_request(r);

        auto [iters, total_tok] = run_until_done(sched2);
        printf("  Policy=%-10s  iters=%d  total_tokens=%d  preemptions=%d\n",
               policy == PreemptionPolicy::SWAP ? "swap" : "recompute",
               iters, total_tok, sched2.total_preemptions());
    }


    // ─────────────────────────────────────────────────────────────────────
    // SECTION 3: Priority-aware vs FCFS
    // ─────────────────────────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 3: Priority-Aware vs FCFS\n");
    printf("============================================================\n\n");

    for (bool prio : {false, true}) {
        // 4 high-priority short requests + 4 low-priority long requests
        auto hp = make_requests(4, 0, 50,  50,  20, 20, 10);
        auto lp = make_requests(4, 1, 200, 200, 80, 80,  1);
        for (int i = 0; i < 4; ++i) { lp[i].seq_id += 4; }

        SchedulerConfig cfg3;
        cfg3.max_num_seqs           = 4;
        cfg3.max_num_batched_tokens = 1024;
        cfg3.enable_priority        = prio;

        BlockManager bm3(500, 16);
        Scheduler    sched3(cfg3, bm3);
        for (auto& r : hp) sched3.add_request(r);
        for (auto& r : lp) sched3.add_request(r);
        for (int i = 0; i < 1000 && !sched3.is_idle(); ++i) sched3.step();

        double hp_avg = 0, lp_avg = 0;
        for (auto& s : sched3.finished_seqs()) {
            if (s.seq_id < 4)  hp_avg += s.finish_iteration;
            else               lp_avg += s.finish_iteration;
        }
        hp_avg /= 4; lp_avg /= 4;
        printf("  %-18s  HP_avg=%.1f  LP_avg=%.1f\n",
               prio ? "Priority-aware" : "FCFS", hp_avg, lp_avg);
    }


    // ─────────────────────────────────────────────────────────────────────
    // SECTION 4: Throughput vs max_num_seqs
    // ─────────────────────────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 4: Throughput vs max_num_seqs\n");
    printf("============================================================\n\n");

    printf("  %-16s %-16s %-14s %s\n",
           "max_num_seqs", "tokens/iter", "utilization", "preemptions");
    printf("  %s\n", std::string(58, '-').c_str());

    for (int max_seqs : {4, 8, 16, 32, 64, 128}) {
        std::mt19937 rng(3);
        std::uniform_int_distribution<int> pdist(32, 200);
        std::uniform_int_distribution<int> gdist(30,  80);

        SchedulerConfig cfg4;
        cfg4.max_num_seqs           = max_seqs;
        cfg4.max_num_batched_tokens = 4096;

        BlockManager bm4(3000, 16);
        Scheduler    sched4(cfg4, bm4);

        int seqs_added = 0;
        std::vector<double> tok_hist, util_hist;

        for (int it = 0; it < 200; ++it) {
            if (it % 8 == 0) {
                for (int k = 0; k < 4; ++k) {
                    Sequence s;
                    s.seq_id         = seqs_added++;
                    s.prompt_len     = pdist(rng);
                    s.max_new_tokens = gdist(rng);
                    sched4.add_request(s);
                }
            }
            IterStats s = sched4.step();
            if (it >= 50) {
                tok_hist.push_back(s.total_tokens);
                util_hist.push_back(s.utilization);
            }
        }

        double avg_tok  = std::accumulate(tok_hist.begin(),  tok_hist.end(),  0.0) / tok_hist.size();
        double avg_util = std::accumulate(util_hist.begin(), util_hist.end(), 0.0) / util_hist.size();
        printf("  %-16d %-16.1f %-13.1f%% %d\n",
               max_seqs, avg_tok, avg_util * 100, sched4.total_preemptions());
    }

    printf("\nDone.\n");
    return 0;
}

```

