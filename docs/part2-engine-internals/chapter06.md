# Chapter 6 — PagedAttention and the Block Manager

> *"The operating system solved this problem for RAM forty years ago. We just had to notice."*
> — vLLM team, 2023

---

## 6.0 Why This Chapter Matters

Chapters 4 and 5 showed what goes *inside* a single attention computation.
This chapter zooms out one level and asks: **where does all of that KV data live
in GPU memory, and how do you manage it efficiently across dozens of concurrent
requests?**

The answer — PagedAttention — is arguably the single most impactful inference
optimization introduced since transformer models became production systems.  It
is the architectural core of vLLM, and understanding it in depth unlocks every
scheduling and memory decision that follows in this book.

By the end of this chapter you will be able to:

- Explain why naïve KV-cache allocation wastes 60–80 % of GPU memory.
- Describe the PagedAttention algorithm and its analogy to OS virtual memory.
- Trace a block table for a concrete four-request batch step by step.
- Understand copy-on-write (CoW) for beam-search and parallel sampling.
- Explain prefix caching and compute the resulting hit-rate improvement.
- Read the vLLM `BlockSpaceManager` source and know which method does what.

---

## 6.1 The Fragmentation Problem  `[FOUNDATIONAL]`

### 6.1.1 How KV cache was allocated before PagedAttention

Before vLLM, virtually every inference server allocated KV cache the same
simple way: **one contiguous tensor per request, sized to the model's maximum
sequence length.**

For a model with max context 4096 tokens the allocation looks like:

```
Per-request KV tensor:
  shape  = [2, n_layers, max_seq_len, n_heads, d_head]
           = [2, 32, 4096, 8, 128]    ← LLaMA 3 8B numbers
  dtype  = bfloat16  (2 bytes)
  bytes  = 2 × 32 × 4096 × 8 × 128 × 2 = 536 870 912  ≈ 512 MB
```

A single A100 80 GB GPU has roughly **60 GB usable** after weights.  Divide by
512 MB and you can run at most **117 requests** — but only if every request
actually fills all 4096 positions.

In practice requests arrive with wildly different lengths.  A typical chat
workload might look like:

```
Request  Actual length   Reserved length   Wasted slots
────────────────────────────────────────────────────────
  r1          42             4096            4054  (99 %)
  r2         380             4096            3716  (91 %)
  r3        1100             4096            2996  (73 %)
  r4        3800             4096             296   (7 %)
  ─────────────────────────────────────────────────────
  Total    5322             16384           11062  (67 %)
```

**Two-thirds of all reserved memory is sitting empty, inaccessible to any other
request.**  This is *internal fragmentation* — the same disease that plagued
early OS memory managers.

There is a second, subtler problem: because you do not know how long each
request will run, you must reserve the *maximum possible* length upfront.  If
a user asks a short question you reserve 4096 slots, decode 12 tokens, and
free the tensor — having wasted the allocation for the entire lifetime of the
request.

### 6.1.2 What the OS learned

Operating systems faced exactly this problem with RAM.  The solution —
introduced in the 1960s, universalised by the 1980s — is **virtual memory with
paging**:

- Physical RAM is divided into fixed-size **frames** (typically 4 KB).
- Each process sees a flat virtual address space; the OS maps virtual **pages**
  to physical frames through a **page table**.
- Frames are allocated on demand and can be scattered anywhere in physical RAM.
- A process that needs 10 MB does not require 10 *contiguous* MB of physical
  RAM — it needs 2560 frames that can live anywhere.

The key insight: **logical contiguity is decoupled from physical contiguity.**

vLLM applies this idea verbatim to the KV cache.

---

## 6.2 PagedAttention — The Core Algorithm  `[DEEP DIVE]`

### 6.2.1 Definitions

| Term | Meaning |
|------|---------|
| **Block** | A fixed-size chunk of KV slots.  Default `block_size = 16` tokens. |
| **Physical block** | A 16-slot tensor that actually lives in GPU SRAM/HBM. |
| **Logical block** | A logical position in a request's KV sequence. |
| **Block table** | Per-request mapping: `logical_block_idx → physical_block_id`. |
| **Block allocator** | The component that hands out / reclaims physical blocks. |

**Figure 6.1 — PagedAttention Block Layout**

<img src="data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgNjgwIDM4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICAgIHN0eWxlPSJtYXgtd2lkdGg6NzAwcHg7Zm9udC1mYW1pbHk6R2VvcmdpYSxzZXJpZjtmb250LXNpemU6MTJweDtkaXNwbGF5OmJsb2NrIj4KICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFycjYiIG1hcmtlcldpZHRoPSI4IiBtYXJrZXJIZWlnaHQ9IjgiIHJlZlg9IjQiIHJlZlk9IjQiIG9yaWVudD0iYXV0byI+CiAgICAgIDxwYXRoIGQ9Ik0xLDEgTDcsNCBMMSw3IHoiIGZpbGw9IiMzNzQxNTEiLz4KICAgIDwvbWFya2VyPgogIDwvZGVmcz4KICA8cmVjdCB3aWR0aD0iNjgwIiBoZWlnaHQ9IjM4MCIgZmlsbD0iI2Y5ZmFmYiIgcng9IjYiIHN0cm9rZT0iI2U1ZTdlYiIvPgoKICA8IS0tIEdQVSBQaHlzaWNhbCBCbG9jayBQb29sIC0tPgogIDxyZWN0IHg9IjE1IiB5PSI0NSIgd2lkdGg9IjMwNSIgaGVpZ2h0PSIzMDUiIHJ4PSI2IiBmaWxsPSIjZWZmNmZmIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPHRleHQgeD0iMTY3IiB5PSIzOCIgZmlsbD0iIzFlNDBhZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMyIgZm9udC13ZWlnaHQ9ImJvbGQiPlBoeXNpY2FsIEtWIEJsb2NrIFBvb2wgKEdQVSk8L3RleHQ+CgogIDwhLS0gQmxvY2sgZ3JpZDogNCBjb2x1bW5zIMOXIDQgcm93cyAtLT4KICA8IS0tIFJvdyAxOiBSZXEgQSBibG9ja3MgKGJsdWUpIC0tPgogIDxyZWN0IHg9IjMwIiAgeT0iNjAiIHdpZHRoPSI2MCIgaGVpZ2h0PSI1NSIgcng9IjMiIGZpbGw9IiNiZmRiZmUiIHN0cm9rZT0iIzI1NjNlYiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KICA8dGV4dCB4PSI2MCIgIHk9IjgzIiAgZmlsbD0iIzFlM2E4YSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMCIgZm9udC13ZWlnaHQ9ImJvbGQiPkJsb2NrIDA8L3RleHQ+CiAgPHRleHQgeD0iNjAiICB5PSI5OSIgIGZpbGw9IiMxZTNhOGEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+UmVxIEE8L3RleHQ+CiAgPHRleHQgeD0iNjAiICB5PSIxMTEiIGZpbGw9IiM2YjcyODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOCI+dG9rcyAwLTE1PC90ZXh0PgoKICA8cmVjdCB4PSIxMDAiIHk9IjYwIiB3aWR0aD0iNjAiIGhlaWdodD0iNTUiIHJ4PSIzIiBmaWxsPSIjYmZkYmZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPHRleHQgeD0iMTMwIiB5PSI4MyIgIGZpbGw9IiMxZTNhOGEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiIGZvbnQtd2VpZ2h0PSJib2xkIj5CbG9jayAxPC90ZXh0PgogIDx0ZXh0IHg9IjEzMCIgeT0iOTkiICBmaWxsPSIjMWUzYThhIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPlJlcSBBPC90ZXh0PgogIDx0ZXh0IHg9IjEzMCIgeT0iMTExIiBmaWxsPSIjNmI3MjgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjgiPnRva3MgMTYtMzE8L3RleHQ+CgogIDwhLS0gUm93IDE6IFJlcSBCIGJsb2NrcyAoYW1iZXIpIC0tPgogIDxyZWN0IHg9IjE3MCIgeT0iNjAiIHdpZHRoPSI2MCIgaGVpZ2h0PSI1NSIgcng9IjMiIGZpbGw9IiNmZWYzYzciIHN0cm9rZT0iI2Q5NzcwNiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KICA8dGV4dCB4PSIyMDAiIHk9IjgzIiAgZmlsbD0iIzc4MzUwZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMCIgZm9udC13ZWlnaHQ9ImJvbGQiPkJsb2NrIDI8L3RleHQ+CiAgPHRleHQgeD0iMjAwIiB5PSI5OSIgIGZpbGw9IiM3ODM1MGYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+UmVxIEI8L3RleHQ+CiAgPHRleHQgeD0iMjAwIiB5PSIxMTEiIGZpbGw9IiM2YjcyODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOCI+dG9rcyAwLTE1PC90ZXh0PgoKICA8IS0tIFJvdyAxOiBSZXEgQyBibG9ja3MgKGdyZWVuKSAtLT4KICA8cmVjdCB4PSIyNDAiIHk9IjYwIiB3aWR0aD0iNjAiIGhlaWdodD0iNTUiIHJ4PSIzIiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPHRleHQgeD0iMjcwIiB5PSI4MyIgIGZpbGw9IiMxNjY1MzQiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiIGZvbnQtd2VpZ2h0PSJib2xkIj5CbG9jayAzPC90ZXh0PgogIDx0ZXh0IHg9IjI3MCIgeT0iOTkiICBmaWxsPSIjMTY2NTM0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPlJlcSBDPC90ZXh0PgogIDx0ZXh0IHg9IjI3MCIgeT0iMTExIiBmaWxsPSIjNmI3MjgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjgiPnRva3MgMC0xNTwvdGV4dD4KCiAgPCEtLSBSb3cgMiAtLT4KICA8cmVjdCB4PSIzMCIgIHk9IjEyNSIgd2lkdGg9IjYwIiBoZWlnaHQ9IjU1IiByeD0iMyIgZmlsbD0iI2JmZGJmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDx0ZXh0IHg9IjYwIiAgeT0iMTQ4IiBmaWxsPSIjMWUzYThhIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+QmxvY2sgNDwvdGV4dD4KICA8dGV4dCB4PSI2MCIgIHk9IjE2NCIgZmlsbD0iIzFlM2E4YSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI5Ij5SZXEgQTwvdGV4dD4KICA8dGV4dCB4PSI2MCIgIHk9IjE3NiIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI4Ij50b2tzIDMyLTQ3PC90ZXh0PgoKICA8cmVjdCB4PSIxMDAiIHk9IjEyNSIgd2lkdGg9IjYwIiBoZWlnaHQ9IjU1IiByeD0iMyIgZmlsbD0iI2ZlZjNjNyIgc3Ryb2tlPSIjZDk3NzA2IiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDx0ZXh0IHg9IjEzMCIgeT0iMTQ4IiBmaWxsPSIjNzgzNTBmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+QmxvY2sgNTwvdGV4dD4KICA8dGV4dCB4PSIxMzAiIHk9IjE2NCIgZmlsbD0iIzc4MzUwZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI5Ij5SZXEgQjwvdGV4dD4KICA8dGV4dCB4PSIxMzAiIHk9IjE3NiIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI4Ij50b2tzIDE2LTMxPC90ZXh0PgoKICA8cmVjdCB4PSIxNzAiIHk9IjEyNSIgd2lkdGg9IjYwIiBoZWlnaHQ9IjU1IiByeD0iMyIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDx0ZXh0IHg9IjIwMCIgeT0iMTQ4IiBmaWxsPSIjMTY2NTM0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+QmxvY2sgNjwvdGV4dD4KICA8dGV4dCB4PSIyMDAiIHk9IjE2NCIgZmlsbD0iIzE2NjUzNCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI5Ij5SZXEgQzwvdGV4dD4KICA8dGV4dCB4PSIyMDAiIHk9IjE3NiIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI4Ij50b2tzIDE2LTMxPC90ZXh0PgoKICA8cmVjdCB4PSIyNDAiIHk9IjEyNSIgd2lkdGg9IjYwIiBoZWlnaHQ9IjU1IiByeD0iMyIgZmlsbD0iI2YzZjRmNiIgc3Ryb2tlPSIjOWNhM2FmIiBzdHJva2Utd2lkdGg9IjEiIHN0cm9rZS1kYXNoYXJyYXk9IjQsMiIvPgogIDx0ZXh0IHg9IjI3MCIgeT0iMTU4IiBmaWxsPSIjOWNhM2FmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIj5GcmVlPC90ZXh0PgoKICA8IS0tIFJvdyAzIC0tPgogIDxyZWN0IHg9IjMwIiAgeT0iMTkwIiB3aWR0aD0iNjAiIGhlaWdodD0iNTUiIHJ4PSIzIiBmaWxsPSIjZjNmNGY2IiBzdHJva2U9IiM5Y2EzYWYiIHN0cm9rZS13aWR0aD0iMSIgc3Ryb2tlLWRhc2hhcnJheT0iNCwyIi8+CiAgPHRleHQgeD0iNjAiICB5PSIyMjMiIGZpbGw9IiM5Y2EzYWYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiPkZyZWU8L3RleHQ+CgogIDxyZWN0IHg9IjEwMCIgeT0iMTkwIiB3aWR0aD0iNjAiIGhlaWdodD0iNTUiIHJ4PSIzIiBmaWxsPSIjZjNmNGY2IiBzdHJva2U9IiM5Y2EzYWYiIHN0cm9rZS13aWR0aD0iMSIgc3Ryb2tlLWRhc2hhcnJheT0iNCwyIi8+CiAgPHRleHQgeD0iMTMwIiB5PSIyMjMiIGZpbGw9IiM5Y2EzYWYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiPkZyZWU8L3RleHQ+CgogIDwhLS0gQ29XIGJsb2NrIChzaGFyZWQgcHJlZml4KSAtLT4KICA8cmVjdCB4PSIxNzAiIHk9IjE5MCIgd2lkdGg9IjYwIiBoZWlnaHQ9IjU1IiByeD0iMyIgZmlsbD0iI2ZjZTdmMyIgc3Ryb2tlPSIjZGIyNzc3IiBzdHJva2Utd2lkdGg9IjIiLz4KICA8dGV4dCB4PSIyMDAiIHk9IjIxMCIgZmlsbD0iIzlkMTc0ZCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI5IiBmb250LXdlaWdodD0iYm9sZCI+QmxvY2sgODwvdGV4dD4KICA8dGV4dCB4PSIyMDAiIHk9IjIyNCIgZmlsbD0iIzlkMTc0ZCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI4Ij5TaGFyZWQgcGZ4PC90ZXh0PgogIDx0ZXh0IHg9IjIwMCIgeT0iMjM4IiBmaWxsPSIjZGIyNzc3IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjgiPkNvVyByZWY9MjwvdGV4dD4KCiAgPHJlY3QgeD0iMjQwIiB5PSIxOTAiIHdpZHRoPSI2MCIgaGVpZ2h0PSI1NSIgcng9IjMiIGZpbGw9IiNmM2Y0ZjYiIHN0cm9rZT0iIzljYTNhZiIgc3Ryb2tlLXdpZHRoPSIxIiBzdHJva2UtZGFzaGFycmF5PSI0LDIiLz4KICA8dGV4dCB4PSIyNzAiIHk9IjIyMyIgZmlsbD0iIzljYTNhZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMCI+RnJlZTwvdGV4dD4KCiAgPCEtLSBCbG9jayB1c2FnZSBiYXIgLS0+CiAgPHRleHQgeD0iMTY3IiB5PSIyNzgiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiPlVzZWQ6IDggLyAxMiBibG9ja3MgKDY3JSk8L3RleHQ+CiAgPHJlY3QgeD0iMzAiIHk9IjI4MyIgd2lkdGg9IjI3MCIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNlNWU3ZWIiLz4KICA8cmVjdCB4PSIzMCIgeT0iMjgzIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjEyIiByeD0iMyIgZmlsbD0iIzI1NjNlYiIgb3BhY2l0eT0iMC42Ii8+CgogIDwhLS0gTGVnZW5kIC0tPgogIDxyZWN0IHg9IjMwIiB5PSIzMDgiIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiIgcng9IjIiIGZpbGw9IiNiZmRiZmUiIHN0cm9rZT0iIzI1NjNlYiIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgPHRleHQgeD0iNDciIHk9IjMxOSIgZmlsbD0iIzM3NDE1MSIgZm9udC1zaXplPSI5Ij5SZXEgQTwvdGV4dD4KICA8cmVjdCB4PSI5MCIgeT0iMzA4IiB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHJ4PSIyIiBmaWxsPSIjZmVmM2M3IiBzdHJva2U9IiNkOTc3MDYiIHN0cm9rZS13aWR0aD0iMSIvPgogIDx0ZXh0IHg9IjEwNyIgeT0iMzE5IiBmaWxsPSIjMzc0MTUxIiBmb250LXNpemU9IjkiPlJlcSBCPC90ZXh0PgogIDxyZWN0IHg9IjE1MCIgeT0iMzA4IiB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHJ4PSIyIiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMSIvPgogIDx0ZXh0IHg9IjE2NyIgeT0iMzE5IiBmaWxsPSIjMzc0MTUxIiBmb250LXNpemU9IjkiPlJlcSBDPC90ZXh0PgogIDxyZWN0IHg9IjIxMCIgeT0iMzA4IiB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHJ4PSIyIiBmaWxsPSIjZmNlN2YzIiBzdHJva2U9IiNkYjI3NzciIHN0cm9rZS13aWR0aD0iMiIvPgogIDx0ZXh0IHg9IjIyNyIgeT0iMzE5IiBmaWxsPSIjMzc0MTUxIiBmb250LXNpemU9IjkiPkNvVyBzaGFyZWQ8L3RleHQ+CgogIDwhLS0gQmxvY2sgVGFibGVzIChyaWdodCkgLS0+CiAgPHJlY3QgeD0iMzQwIiB5PSI0NSIgd2lkdGg9IjMyMCIgaGVpZ2h0PSIzMDUiIHJ4PSI2IiBmaWxsPSIjZmFmYWZhIiBzdHJva2U9IiNkMWQ1ZGIiIHN0cm9rZS13aWR0aD0iMSIvPgogIDx0ZXh0IHg9IjUwMCIgeT0iMzgiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTMiIGZvbnQtd2VpZ2h0PSJib2xkIj5Mb2dpY2FsIOKGkiBQaHlzaWNhbCBCbG9jayBUYWJsZXM8L3RleHQ+CgogIDwhLS0gUmVxIEEgdGFibGUgLS0+CiAgPHRleHQgeD0iMzgwIiB5PSI3NSIgZmlsbD0iIzFlNDBhZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9ImJvbGQiPlJlcSBBICgzIGJsb2Nrcyk8L3RleHQ+CiAgPHJlY3QgeD0iMzU1IiB5PSI4MiIgd2lkdGg9IjEyMCIgaGVpZ2h0PSIyMCIgcng9IjIiIGZpbGw9IiNiZmRiZmUiLz4KICA8dGV4dCB4PSIzNjUiIHk9Ijk2IiBmaWxsPSIjMWUzYThhIiBmb250LXNpemU9IjkiPkxvZ2ljYWwgMCDihpIgQmxvY2sgMDwvdGV4dD4KICA8cmVjdCB4PSIzNTUiIHk9IjEwNCIgd2lkdGg9IjEyMCIgaGVpZ2h0PSIyMCIgcng9IjIiIGZpbGw9IiNiZmRiZmUiIG9wYWNpdHk9IjAuNyIvPgogIDx0ZXh0IHg9IjM2NSIgeT0iMTE4IiBmaWxsPSIjMWUzYThhIiBmb250LXNpemU9IjkiPkxvZ2ljYWwgMSDihpIgQmxvY2sgMTwvdGV4dD4KICA8cmVjdCB4PSIzNTUiIHk9IjEyNiIgd2lkdGg9IjEyMCIgaGVpZ2h0PSIyMCIgcng9IjIiIGZpbGw9IiNiZmRiZmUiIG9wYWNpdHk9IjAuNSIvPgogIDx0ZXh0IHg9IjM2NSIgeT0iMTQwIiBmaWxsPSIjMWUzYThhIiBmb250LXNpemU9IjkiPkxvZ2ljYWwgMiDihpIgQmxvY2sgNDwvdGV4dD4KCiAgPCEtLSBSZXEgQiB0YWJsZSAtLT4KICA8dGV4dCB4PSIzODAiIHk9IjE2OCIgZmlsbD0iIzkyNDAwZSIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9ImJvbGQiPlJlcSBCICgyIGJsb2Nrcyk8L3RleHQ+CiAgPHJlY3QgeD0iMzU1IiB5PSIxNzUiIHdpZHRoPSIxMjAiIGhlaWdodD0iMjAiIHJ4PSIyIiBmaWxsPSIjZmVmM2M3Ii8+CiAgPHRleHQgeD0iMzY1IiB5PSIxODkiIGZpbGw9IiM3ODM1MGYiIGZvbnQtc2l6ZT0iOSI+TG9naWNhbCAwIOKGkiBCbG9jayAyPC90ZXh0PgogIDxyZWN0IHg9IjM1NSIgeT0iMTk3IiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjIwIiByeD0iMiIgZmlsbD0iI2ZlZjNjNyIgb3BhY2l0eT0iMC43Ii8+CiAgPHRleHQgeD0iMzY1IiB5PSIyMTEiIGZpbGw9IiM3ODM1MGYiIGZvbnQtc2l6ZT0iOSI+TG9naWNhbCAxIOKGkiBCbG9jayA1PC90ZXh0PgoKICA8IS0tIFJlcSBDIHRhYmxlIC0tPgogIDx0ZXh0IHg9IjM4MCIgeT0iMjM0IiBmaWxsPSIjMTY2NTM0IiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iYm9sZCI+UmVxIEMgKDIgYmxvY2tzKTwvdGV4dD4KICA8cmVjdCB4PSIzNTUiIHk9IjI0MSIgd2lkdGg9IjEyMCIgaGVpZ2h0PSIyMCIgcng9IjIiIGZpbGw9IiNkY2ZjZTciLz4KICA8dGV4dCB4PSIzNjUiIHk9IjI1NSIgZmlsbD0iIzE2NjUzNCIgZm9udC1zaXplPSI5Ij5Mb2dpY2FsIDAg4oaSIEJsb2NrIDM8L3RleHQ+CiAgPHJlY3QgeD0iMzU1IiB5PSIyNjMiIHdpZHRoPSIxMjAiIGhlaWdodD0iMjAiIHJ4PSIyIiBmaWxsPSIjZGNmY2U3IiBvcGFjaXR5PSIwLjciLz4KICA8dGV4dCB4PSIzNjUiIHk9IjI3NyIgZmlsbD0iIzE2NjUzNCIgZm9udC1zaXplPSI5Ij5Mb2dpY2FsIDEg4oaSIEJsb2NrIDY8L3RleHQ+CgogIDwhLS0gS2V5IHBvaW50IGJveCAtLT4KICA8cmVjdCB4PSI0OTAiIHk9Ijc1IiB3aWR0aD0iMTU1IiBoZWlnaHQ9IjEyMCIgcng9IjQiIGZpbGw9IiNmZmY3ZWQiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgPHRleHQgeD0iNTY3IiB5PSI5MyIgZmlsbD0iI2MyNDEwYyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMCIgZm9udC13ZWlnaHQ9ImJvbGQiPktleSBJbnNpZ2h0PC90ZXh0PgogIDx0ZXh0IHg9IjU2NyIgeT0iMTEwIiBmaWxsPSIjMzc0MTUxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPlBoeXNpY2FsIGJsb2NrcyBhcmU8L3RleHQ+CiAgPHRleHQgeD0iNTY3IiB5PSIxMjQiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+bm9uLWNvbnRpZ3VvdXMuPC90ZXh0PgogIDx0ZXh0IHg9IjU2NyIgeT0iMTQwIiBmaWxsPSIjMzc0MTUxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPkJsb2NrIHRhYmxlIG1hcHM8L3RleHQ+CiAgPHRleHQgeD0iNTY3IiB5PSIxNTQiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+bG9naWNhbCDihpIgcGh5c2ljYWw8L3RleHQ+CiAgPHRleHQgeD0iNTY3IiB5PSIxNzAiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+YXQgZGVjb2RlIHRpbWUuPC90ZXh0PgoKICA8cmVjdCB4PSI0OTAiIHk9IjIxMCIgd2lkdGg9IjE1NSIgaGVpZ2h0PSI4MCIgcng9IjQiIGZpbGw9IiNmY2U3ZjMiIHN0cm9rZT0iI2RiMjc3NyIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgPHRleHQgeD0iNTY3IiB5PSIyMjgiIGZpbGw9IiM5ZDE3NGQiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiIGZvbnQtd2VpZ2h0PSJib2xkIj5Db3B5LW9uLVdyaXRlPC90ZXh0PgogIDx0ZXh0IHg9IjU2NyIgeT0iMjQ0IiBmaWxsPSIjMzc0MTUxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPkJlYW0gc2VhcmNoIGZvcmtzPC90ZXh0PgogIDx0ZXh0IHg9IjU2NyIgeT0iMjU4IiBmaWxsPSIjMzc0MTUxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPnNoYXJlIHByZWZpeCBibG9ja3M8L3RleHQ+CiAgPHRleHQgeD0iNTY3IiB5PSIyNzIiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+dW50aWwgZmlyc3Qgd3JpdGUuPC90ZXh0PgoKICA8IS0tIGFycm93cyBmcm9tIHRhYmxlIHRvIGJsb2NrcyAoZG90dGVkKSAtLT4KICA8bGluZSB4MT0iNDc1IiB5MT0iOTIiICB4Mj0iMzEwIiB5Mj0iODciICBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMC44IiBzdHJva2UtZGFzaGFycmF5PSIzLDIiIG1hcmtlci1lbmQ9InVybCgjYXJyNikiLz4KICA8bGluZSB4MT0iNDc1IiB5MT0iMTg2IiB4Mj0iMzEwIiB5Mj0iMTUyIiBzdHJva2U9IiNkOTc3MDYiIHN0cm9rZS13aWR0aD0iMC44IiBzdHJva2UtZGFzaGFycmF5PSIzLDIiIG1hcmtlci1lbmQ9InVybCgjYXJyNikiLz4KICA8bGluZSB4MT0iNDc1IiB5MT0iMjUwIiB4Mj0iMzEwIiB5Mj0iMjE3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMC44IiBzdHJva2UtZGFzaGFycmF5PSIzLDIiIG1hcmtlci1lbmQ9InVybCgjYXJyNikiLz4KPC9zdmc+" style="max-width:700px;width:100%;display:block;margin:1rem 0" alt="diagram"/>

For LLaMA 3 8B with `block_size = 16`:

```
One physical block:
  shape  = [2, n_layers, block_size, n_heads, d_head]
           = [2, 32, 16, 8, 128]
  dtype  = bfloat16
  bytes  = 2 × 32 × 16 × 8 × 128 × 2 = 2 097 152  ≈ 2 MB
```

An A100 with 60 GB usable for KV can hold:

```
  60 × 10⁹ / 2 097 152 ≈ 28 600 physical blocks
  28 600 × 16 = 457 600 total KV slots
```

### 6.2.2 Allocation on demand

When a new request arrives the block manager allocates **only the blocks needed
for the current tokens**.  As the request generates new tokens, additional blocks
are allocated one at a time — never more than one block ahead.

```
Step 0 (prefill, 5 tokens):
  logical blocks needed = ceil(5/16) = 1
  physical blocks allocated: [phys_7]

  Block table for r1:
  ┌─────────────┬──────────────┐
  │ logical blk │ physical blk │
  ├─────────────┼──────────────┤
  │      0      │      7       │  ← tokens 0-4 here, slots 5-15 free
  └─────────────┴──────────────┘

Step 1-10 (10 decode steps):
  Still in block 0.  No new allocation.

Step 11 (token 16 — first token of second block):
  Allocate new block [phys_23]
  ┌─────────────┬──────────────┐
  │      0      │      7       │  ← full (tokens 0-15)
  │      1      │     23       │  ← token 16 here
  └─────────────┴──────────────┘
```

No memory is reserved for future tokens that may never arrive.

### 6.2.3 Attention over non-contiguous blocks

Standard attention (Chapter 4) assumed all K and V tensors are contiguous.
PagedAttention requires a small kernel change: instead of a single pointer
to a contiguous KV buffer, the kernel receives the **block table** and fetches
tiles from scattered physical locations.

The computation is mathematically identical:

```
For query q_i (token position i):
  1. Look up block_table[request_id] to get list of physical block ids.
  2. For each physical block b:
       - Load K[b]  (16 key   vectors, d_k dims)
       - Load V[b]  (16 value vectors, d_v dims)
       - Compute partial scores  s_partial = q_i · K[b]ᵀ / √d_k
       - Accumulate into online softmax running state (m, ℓ, O)
         (same merge rule as Flash Attention, Chapter 5)
  3. Finalise: O_i = O / ℓ
```

The kernel is essentially Flash Attention's tiling loop with the K/V pointer
replaced by a table lookup.  FlashInfer (Chapter 5 §5.10) implements this
natively as its `paged_attention` kernel.

---

## 6.3 Worked Example — Four-Request Batch  `[DEEP DIVE]`

We trace a small system: `block_size = 4`, `total_blocks = 12` (IDs 0–11),
three requests arriving at different times.

### 6.3.1 Initial state

```
Physical block pool (free):
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Block tables: (empty)
```

### 6.3.2 Step A — r1 arrives (prefill 7 tokens)

```
Blocks needed = ceil(7/4) = 2
Allocate: phys_0, phys_1

Block table r1:
  logical 0 → phys_0   (tokens 0-3, all filled)
  logical 1 → phys_1   (tokens 4-6, slot 7 free)

Free pool: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

### 6.3.3 Step B — r2 arrives (prefill 5 tokens)

```
Blocks needed = ceil(5/4) = 2
Allocate: phys_2, phys_3

Block table r2:
  logical 0 → phys_2   (tokens 0-3, all filled)
  logical 1 → phys_3   (tokens 4, slots 5-7 free)

Free pool: [4, 5, 6, 7, 8, 9, 10, 11]
```

### 6.3.4 Step C — Decode step (r1 generates token 8, r2 generates token 6)

```
r1: token 8 fits in phys_1 (slot 4).  No new block.
r2: token 6 fits in phys_3 (slot 5).  No new block.

Block table r1:  logical 0 → phys_0,  logical 1 → phys_1  (slots 0-4 used)
Block table r2:  logical 0 → phys_2,  logical 1 → phys_3  (slots 0-5 used)
```

### 6.3.5 Step D — r3 arrives (prefill 3 tokens), r1 generates token 9

```
r3: blocks needed = 1.  Allocate phys_4.
r1: token 9 fits in phys_1 (slot 5).  No new block.

Block table r3:  logical 0 → phys_4   (tokens 0-2, slot 3 free)

Free pool: [5, 6, 7, 8, 9, 10, 11]
```

### 6.3.6 Step E — r2 completes (frees blocks), r1 generates token 12

```
r2 done → free phys_2, phys_3.  Free pool: [2, 3, 5, 6, 7, 8, 9, 10, 11]

r1: token 12 falls at position 12 = block index ceil(12/4)=3.
    Actually position 12 means slot 0 of logical block 2 (0-indexed: positions
    0-3 in blk 0, 4-7 in blk 1, 8-11 in blk 2, 12-15 in blk 3).
    Wait — let me recalculate: r1 started with 7 tokens (0-6), generated tokens
    at positions 7,8,9 in step C/D above. Token at position 12 → block
    index = 12 // 4 = 3.  Allocate phys_2 (first free).

Block table r1:
  logical 0 → phys_0   (tokens  0- 3)
  logical 1 → phys_1   (tokens  4- 7)
  logical 2 → phys_5   (tokens  8-11)   ← allocated in step D above
  logical 3 → phys_2   (token  12, slots 13-15 free)

Free pool: [3, 6, 7, 8, 9, 10, 11]
```

**Key observation**: phys_2 was freed by r2 and immediately reused by r1.
In the naïve scheme r2's tensor could not be reclaimed until the entire request
completed; here each 4-slot block is returned the moment it is no longer needed.

### 6.3.7 ASCII memory map after Step E

```
Physical block layout (each cell = 4 KV slots):

  phys_0  │ r1 tokens  0- 3 │  (full)
  phys_1  │ r1 tokens  4- 7 │  (full)
  phys_2  │ r1 token  12    │  (1/4 full)
  phys_3  │ FREE            │
  phys_4  │ r3 tokens  0- 2 │  (3/4 full)
  phys_5  │ r1 tokens  8-11 │  (full)
  phys_6  │ FREE            │
  phys_7  │ FREE            │
  ...
  phys_11 │ FREE            │

r1 block table: [0→0, 1→1, 2→5, 3→2]   (logical → physical)
r3 block table: [0→4]
```

The physical blocks for r1 are scattered (IDs 0, 1, 5, 2) — but r1's attention
kernel doesn't care, because the block table provides the mapping.

### 6.3.8 Memory utilization comparison

```
Naïve (max-len reservation, max_len=16):

  r1 reserved: 16 slots.  Used: 13.  Wasted: 3 / 16 = 18.75 %
  r2 reserved: 16 slots.  Used:  6.  Wasted: 10 / 16 = 62.5 %
  r3 reserved: 16 slots.  Used:  3.  Wasted: 13 / 16 = 81.25 %
  Total wasted: 26 / 48 slots = 54 %

PagedAttention (block_size=4):

  r1 allocated:  4 blocks × 4 = 16 slots.  Used: 13.  Internal waste: 3/16 = 18.75 %
  r3 allocated:  1 block  × 4 =  4 slots.  Used:  3.  Internal waste:  1/4 = 25 %
  r2 already freed its blocks (0 wasted after completion).
  Total allocated: 5 blocks.  Used: 16.  Internal waste: 4/20 = 20 %
```

PagedAttention reduces *peak* memory waste from 54 % to 20 % in this example,
and the freed blocks from completed requests are immediately reusable.

---

## 6.4 The Block Manager Implementation  `[DEEP DIVE]`

### 6.4.1 Data structures

In the vLLM source (`vllm/core/block_manager.py`) the core structures are:

```python
# Simplified version of vLLM's actual implementation

class PhysicalBlock:
    block_id:   int          # GPU memory index
    ref_count:  int = 0      # how many sequences share this block
    block_hash: int = None   # for prefix caching (§6.6)
    num_hashed_tokens: int = 0

class BlockAllocator:
    """Manages a pool of physical blocks (GPU or CPU)."""

    def __init__(self, device: str, num_blocks: int, block_size: int):
        self.free_blocks: List[PhysicalBlock] = [
            PhysicalBlock(i) for i in range(num_blocks)
        ]
        self.device = device
        self.block_size = block_size
        # For prefix caching:
        self.cached_blocks: Dict[int, PhysicalBlock] = {}

    def allocate(self, block_hash=None, num_hashed_tokens=0) -> PhysicalBlock:
        if not self.free_blocks:
            raise BlockAllocatorError("Out of GPU memory blocks")
        block = self.free_blocks.pop()
        block.ref_count = 1
        block.block_hash = block_hash
        block.num_hashed_tokens = num_hashed_tokens
        return block

    def free(self, block: PhysicalBlock) -> None:
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)
```

The per-request view is a `BlockTable` — a list of `PhysicalBlock` objects:

```python
class BlockTable:
    """Maps logical block indices to physical blocks for one sequence."""

    def __init__(self, block_size: int, allocator: BlockAllocator):
        self._blocks: List[PhysicalBlock] = []
        self.block_size = block_size
        self._allocator = allocator

    def allocate(self, token_ids: List[int]) -> None:
        """Allocate blocks for an initial token list (prefill)."""
        n_blocks = (len(token_ids) + self.block_size - 1) // self.block_size
        for _ in range(n_blocks):
            self._blocks.append(self._allocator.allocate())

    def append_slot(self) -> Optional[int]:
        """
        Ensure there is space for one more token.
        Returns the physical block id if a new block was just allocated,
        None if the current last block has room.
        """
        last_block = self._blocks[-1]
        tokens_in_last = (self.num_filled_slots - 1) % self.block_size + 1
        if tokens_in_last < self.block_size:
            return None                   # room in current block
        new_block = self._allocator.allocate()
        self._blocks.append(new_block)
        return new_block.block_id

    @property
    def physical_block_ids(self) -> List[int]:
        return [b.block_id for b in self._blocks]
```

### 6.4.2 The BlockSpaceManager

`BlockSpaceManager` (vLLM's actual class name) sits above the allocator and
handles the lifecycle for a whole batch:

```
can_allocate(seq_group)   → NEVER / MAY / OK
allocate(seq_group)       → calls block_table.allocate() for prefill
can_append_slot(seq_group)→ check ≥1 free block
append_slot(seq, block_table) → calls block_table.append_slot()
free(seq)                 → returns all blocks to allocator
swap_out(seq_group)       → move GPU blocks → CPU blocks
swap_in(seq_group)        → move CPU blocks → GPU blocks
```

The scheduler (Chapter 8) calls these methods at every iteration to decide
which requests to run, preempt, or swap.

### 6.4.3 `[COMMON TRAP]` — Off-by-one in slot counting

A classic bug: when does `append_slot` allocate a new block?

```
block_size = 4.
After prefill of 4 tokens: 1 block, all 4 slots filled.
Next token (decode step 1): append_slot MUST allocate a new block.
After prefill of 5 tokens: 2 blocks; block 0 full, block 1 has 1 slot.
Next token (decode step 1): fits in block 1's slot 2.  No allocation.
```

The condition is: `(num_tokens % block_size) == 0` after incrementing.
In Python: `if len(token_ids) % block_size == 0: allocate_new_block()`.

Getting this wrong by 1 either wastes a block per request or causes a
buffer overflow into another request's memory.

---

## 6.5 Copy-on-Write for Beam Search and Parallel Sampling  `[DEEP DIVE]`

### 6.5.1 The sharing problem

Beam search maintains B candidate sequences (the "beams").  At every step each
beam generates the next token and the top-B candidates across all beams are kept.

Naïvely, B beams require B full KV cache copies — even though all beams share
the same prompt and much of their history.

PagedAttention solves this with **block sharing + copy-on-write**:

```
Prompt KV (shared):  logical block 0 → phys_7   (ref_count = B)
                     logical block 1 → phys_12  (ref_count = B)

Beam 0:  [...phys_7, phys_12, phys_31]   (phys_31 is beam 0's private block)
Beam 1:  [...phys_7, phys_12, phys_44]
Beam 2:  [...phys_7, phys_12, phys_55]
```

The shared blocks (phys_7, phys_12) have `ref_count = 3`.  They are *not*
copied unless a beam needs to write into them.

### 6.5.2 Copy-on-write protocol

When does a beam write into a shared block?  Only when it needs to *modify*
a token that was already written — which in autoregressive decoding never
happens for past tokens (you never rewrite a KV entry once computed).

The only write operation in decode mode is *appending* the new token's KV to
the last (partially filled) block.  If that last block is shared (`ref_count > 1`)
the block manager performs a CoW:

```
1. Allocate a new physical block  phys_new.
2. Copy contents of shared block → phys_new.
3. Update beam's block table: last logical block → phys_new.
4. Decrement ref_count of old block.
5. Set ref_count of phys_new = 1.
6. Write new token's KV into phys_new.
```

```
Before CoW (beam 0 about to extend, sharing phys_12):

  Beam 0 block table:  [phys_7, phys_12]   ref_count(phys_12) = 3

After CoW:

  Beam 0 block table:  [phys_7, phys_99]   ref_count(phys_12) = 2
                                            ref_count(phys_99) = 1

  Beam 1 still sees:   [phys_7, phys_12]
  Beam 2 still sees:   [phys_7, phys_12]
```

### 6.5.3 Memory savings from sharing

For a beam search with B=4, sequence length 512, block_size=16:

```
  Blocks per sequence: 512 / 16 = 32

  Without sharing:   4 × 32 = 128 blocks
  With CoW sharing:
    Shared prefix (say, 256 tokens = 16 blocks):  16 blocks × ref_count=4
    Private suffix   (256 tokens = 16 blocks):    4 × 16 = 64 blocks
    Total physical blocks:  16 + 64 = 80 blocks

  Saving: (128 - 80) / 128 = 37.5 %
```

For longer shared prefixes (system prompts, RAG contexts) the saving grows
proportionally.

### 6.5.4 ASCII — CoW during beam search

```
After prefill (all beams identical, B=3):
                         ┌─────────────────────────────────────┐
  Beam 0 block table:    │ phys_0 | phys_1 | phys_2           │
  Beam 1 block table:    │ phys_0 | phys_1 | phys_2           │  ref_count = 3
  Beam 2 block table:    │ phys_0 | phys_1 | phys_2           │
                         └─────────────────────────────────────┘
                           (shared prefix)      (last partial blk, shared)

Decode step 1: all beams extend phys_2 (last shared block).
  CoW triggered for each beam in turn:

  Beam 0 first:
    alloc phys_3;  copy phys_2 → phys_3;  beam 0 → phys_3;  ref(phys_2)=2
  Beam 1 next:
    alloc phys_4;  copy phys_2 → phys_4;  beam 1 → phys_4;  ref(phys_2)=1
  Beam 2 last:
    ref(phys_2)=1 already → just write in place, no copy needed.
    ref(phys_2) stays 1.

After decode step 1:
  Beam 0: [phys_0, phys_1, phys_3]
  Beam 1: [phys_0, phys_1, phys_4]
  Beam 2: [phys_0, phys_1, phys_2]
  phys_0, phys_1 still shared (ref=3).
```

---

## 6.6 Prefix Caching  `[DEEP DIVE]`

### 6.6.1 Motivation

In many deployments the same prefix appears in every request:

- A long system prompt ("You are a helpful assistant…")
- A RAG context prepended before every user question
- A few-shot example block

With naïve allocation each request recomputes and re-stores these KV entries.
With prefix caching, a block whose content hash matches an existing cached block
can be **shared read-only**.

### 6.6.2 Block hashing

A physical block is eligible for prefix caching if:
1. It is *full* (all `block_size` slots are written).
2. It is *immutable* (no pending CoW — it has `ref_count ≥ 1` and is not the
   last block of any active sequence).

The block's hash is computed over the token IDs it contains plus the hash of
the preceding block (creating a chain hash, so order matters):

```python
def compute_block_hash(token_ids: List[int], prev_block_hash: int) -> int:
    return hash((tuple(token_ids), prev_block_hash))
```

### 6.6.3 Cache lookup on prefill

When a new request arrives with a long prompt the allocator checks, block by
block, whether a matching cached block exists:

```
New request prompt: "You are a helpful AI. Answer concisely. User: What is 2+2?"

Block 0 (tokens 0-15):  hash → h0.  Found in cache → phys_7 (ref_count++ = 2).
Block 1 (tokens 16-31): hash → h1.  Found in cache → phys_12 (ref_count++ = 2).
Block 2 (tokens 32-39): partial, cannot cache-match.  Allocate fresh phys_33.

KV computation needed: only block 2 (8 tokens).  Blocks 0,1 reused from cache.
```

This is called a **prefix cache hit**.  The TTFT (time-to-first-token) for the
new request drops dramatically because most of the prefill attention is skipped.

### 6.6.4 Cache eviction

When the free pool is empty and a new block is needed, the block manager evicts
the **least-recently-used** cached block (LRU eviction):

```
Eviction candidates: blocks with ref_count == 1 (no active reference)
  and  block_hash is not None (they were cached).

Pick LRU → decrement ref_count to 0 → add to free pool.
```

Blocks actively referenced by running requests are *never* evicted; only
"stale" cached blocks (retained speculatively) are candidates.

**WORKED EXAMPLE 6.6 — Block eviction under memory pressure:**

```
WORKED EXAMPLE 6.6 — Block eviction under memory pressure
──────────────────────────────────────────────────────────────────────────

SETUP
─────
Block pool total capacity : 200 blocks
Block pool used           : 190 blocks  (95% full)
Block pool free           : 10 blocks
Block size                : 16 tokens

4 running sequences (S1–S4) hold actively-referenced blocks:
  S1: 48 tokens generated  → 3 blocks  (last-access t=100)
  S2: 80 tokens generated  → 5 blocks  (last-access t=105)
  S3: 32 tokens generated  → 2 blocks  (last-access t=98)
  S4: 64 tokens generated  → 4 blocks  (last-access t=102)
  Total active blocks      : 3+5+2+4 = 14 blocks

Remaining 176 blocks are stale cached blocks (ref_count==1, have
block_hash), retained speculatively for future prefix-cache hits.
Their last-access timestamps (a representative sample):

  Cache group A (system prompt for workspace app):
    blocks A1–A8   last_access = t=50   (8 blocks, oldest)
  Cache group B (system prompt for coding assistant):
    blocks B1–B6   last_access = t=72   (6 blocks)
  Cache group C (conversation history, user X):
    blocks C1–C4   last_access = t=88   (4 blocks)
  Cache group D (conversation history, user Y):
    blocks D1–D3   last_access = t=95   (3 blocks)
  ... (remaining 155 blocks with t ≥ 96, not targeted yet)

New request R5 arrives with a 400-token prompt → needs 25 blocks
(⌈400/16⌉ = 25).  After prefix-cache lookup, 10 blocks hit (partial
prefix match); 15 new blocks must be allocated.

FREE POOL CHECK
───────────────
  Currently free : 10 blocks
  Needed         : 15 blocks
  Shortfall      :  5 blocks  → must evict at least 5 stale blocks

TOKEN BUDGET BEFORE EVICTION
─────────────────────────────
  Total pool capacity        : 200 blocks
  Active (running sequences) :  14 blocks   ← PROTECTED, never evicted
  Stale cached (evictable)   : 176 blocks
  Free                       :  10 blocks
  Token budget represented   : 200 × 16 = 3,200 tokens total in pool

LRU EVICTION — selecting candidates
────────────────────────────────────
LRU policy selects stale blocks in ascending last_access order.
Only blocks with ref_count==1 AND block_hash != None are candidates.
Actively-referenced blocks (S1–S4) are excluded.

Eviction pass 1 — need 5 more free blocks:

  Step 1: Evict A1 (last_access=50)  → free_pool = 11
  Step 2: Evict A2 (last_access=50)  → free_pool = 12
  Step 3: Evict A3 (last_access=50)  → free_pool = 13
  Step 4: Evict A4 (last_access=50)  → free_pool = 14
  Step 5: Evict A5 (last_access=50)  → free_pool = 15  ✓ target reached

  Blocks evicted: A1–A5 (5 of the 8 oldest system-prompt cache blocks)
  Blocks remaining in cache group A: A6–A8 (last_access=50, still cached)

TOKEN BUDGET AFTER EVICTION
────────────────────────────
  Total pool capacity        : 200 blocks
  Active (running sequences) :  14 blocks   (unchanged, S1–S4 still run)
  Stale cached (evictable)   : 171 blocks   (176 − 5 evicted)
  Free                       :  15 blocks   (10 + 5 freed)
  New request R5 (15 new blocks allocated, 10 from cache hit):
    → R5 now running, consuming 25 blocks
  Free after R5 allocation   :  15 − 15 = 0 blocks

  Post-allocation snapshot:
    Active blocks : 14 (S1–S4) + 25 (R5) = 39 blocks
    Stale cached  : 171 blocks
    Free          :   0 blocks  (pool is now full again)

BLOCK POOL STATE — ASCII DIAGRAM
──────────────────────────────────

  BEFORE eviction + R5 arrival:
  ┌────────────────────────────────────────────────────────────────┐
  │ BLOCK POOL (200 blocks total)                                  │
  │                                                                │
  │  ██████████████  ← S1–S4 active (14 blocks, PROTECTED)        │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  ░  Stale cached blocks (176 blocks, LRU-evictable)  ░░░░░░░  │
  │  ░  [A1–A8: t=50] [B1–B6: t=72] [C1–C4: t=88] ...  ░░░░░░░  │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  □□□□□□□□□□  ← Free pool (10 blocks)                          │
  │                                                                │
  │  Usage: 190/200 = 95.0%                                        │
  └────────────────────────────────────────────────────────────────┘

  AFTER eviction of A1–A5 and allocation to R5:
  ┌────────────────────────────────────────────────────────────────┐
  │ BLOCK POOL (200 blocks total)                                  │
  │                                                                │
  │  ███████████████████████  ← S1–S4 + R5 active (39 blocks)     │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  ░  Stale cached blocks (171 blocks)                  ░░░░░░  │
  │  ░  [A6–A8: t=50] [B1–B6: t=72] [C1–C4: t=88] ...   ░░░░░░  │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  (no free blocks)                                              │
  │                                                                │
  │  Usage: 200/200 = 100.0%  (next allocation triggers eviction)  │
  └────────────────────────────────────────────────────────────────┘

WHAT HAPPENS TO S1–S4?
───────────────────────
Nothing — they are actively running and NEVER touched by LRU eviction.
Only stale cached blocks (ref_count==1) are eviction candidates.

The evicted blocks (A1–A5) belonged to a system-prompt prefix cache,
not to any running sequence.  Effect: the next request that shares that
system prompt will get a partial cache miss on blocks A1–A5 and must
recompute those 5 × 16 = 80 tokens of prefill.  This is the cost of
LRU eviction — it trades future cache-hit savings for immediate memory.

WHAT IF NO STALE BLOCKS EXIST?
────────────────────────────────
If the pool is full of actively-referenced blocks (no evictable stale
blocks), the scheduler must preempt a running sequence:

  Swap path   (default if cpu_offload_gb > 0):
    Pick lowest-priority running sequence (e.g. S3, fewest tokens).
    Copy S3's 2 GPU blocks → CPU DRAM.
    Mark S3 as SWAPPED.  Free 2 GPU blocks.
    If 2 blocks still not enough, preempt S1 (3 blocks) → 5 freed.

  Recompute path (if cpu_offload_gb == 0):
    Discard S3's KV blocks entirely.
    Mark S3 as WAITING.  When GPU memory is available again,
    S3 re-enters RUNNING and recomputes its full KV state from scratch.

  Swap cost example (S3, 2 blocks, PCIe 4.0):
    2 blocks × 2 MB/block = 4 MB to transfer
    4 MB / 32 GB/s ≈ 125 µs swap-out latency
    Recompute cost (32 tokens on LLaMA 3 8B): ~0.5 ms prefill
    → Swap wins for short sequences; recompute may win for long ones
      if the sequence will not resume for many seconds.

SUMMARY
───────
  Blocks evicted  : 5 (A1–A5, last_access=50, stale cached)
  Sequences preempted : 0  (eviction of stale cache was sufficient)
  Free pool after R5 : 0 blocks (next step will trigger further eviction)
  Cache group A hit rate impact: requests sharing that system prompt
    will incur a partial miss until A1–A5 are recomputed and re-cached.
──────────────────────────────────────────────────────────────────────────
```

### 6.6.5 Cache hit rate — worked example

System prompt = 512 tokens → 32 blocks.
Requests arrive one per second for one minute (60 requests).
All share the same system prompt.

```
Request 1:  miss (cache cold).  Computes 32 blocks.  Caches all 32.
Requests 2-60:  32 blocks hit, 0 miss (assuming cache not evicted).

Hit rate = (59 × 32) / (60 × 32)  =  1888 / 1920  ≈ 98.3 %

Compute saved: 59/60 = 98.3 % of system-prompt prefill FLOPs.
```

For a 512-token system prompt on LLaMA 3 8B, the prefill cost is roughly
1.7 × 10¹² FLOPs (see Chapter 2 formula).  Saving 98 % of that over 60 requests
is ~100 × 10¹² FLOPs — equivalent to the full model forward of ~2000 tokens.

---

## 6.7 GPU ↔ CPU Swapping  `[DEEP DIVE]`

### 6.7.1 Why swap?

When the GPU block pool is exhausted and the scheduler cannot preempt a
running request, it can **swap** a waiting request's blocks to CPU DRAM.
The request is paused, its GPU blocks are copied to CPU blocks, and the GPU
blocks are freed.  When the request resumes, blocks are swapped back in.

### 6.7.2 Swap cost

```
Block size: 2 MB (LLaMA 3 8B, block_size=16).
PCIe 4.0 bandwidth: ~32 GB/s bidirectional.
Swap latency per block: 2 MB / 32 GB/s ≈ 62.5 µs.

A sequence with 1024 tokens = 64 blocks:
  Swap-out: 64 × 62.5 µs = 4 ms.
  Swap-in:  4 ms.

Total suspend + resume overhead: ~8 ms.
```

For a decode step taking ~20 ms (Chapter 8) this is significant but acceptable
for low-priority requests.  High-priority requests should never be swapped.

### 6.7.3 CPU block allocator

vLLM maintains a separate `BlockAllocator` for CPU memory, with its own free
list.  `swap_out` calls:

```python
def swap_out(seq_group: SequenceGroup) -> Dict[int, int]:
    """Returns mapping gpu_block_id → cpu_block_id."""
    mapping = {}
    for seq in seq_group.seqs:
        for gpu_block in seq.block_table:
            cpu_block = self.cpu_allocator.allocate()
            # Async copy: GPU → CPU (CUDA memcpy D→H)
            copy_block_gpu_to_cpu(gpu_block.block_id, cpu_block.block_id)
            mapping[gpu_block.block_id] = cpu_block.block_id
            self.gpu_allocator.free(gpu_block)
        seq.block_table = [cpu_block for cpu_block in ...]
    return mapping
```

---

## 6.8 vLLM Scheduler Integration  `[DEEP DIVE]`

### 6.8.1 Decision tree per iteration

```
┌──────────────────────────────────────────────────────────────┐
│                  SCHEDULER ITERATION t                       │
│                                                              │
│  For each waiting request w (in priority order):            │
│    1. can_allocate(w)?                                       │
│       ├─ NEVER      → skip (too large to ever fit)          │
│       ├─ MAY LATER  → stop admitting new requests           │
│       └─ OK         → allocate(w), move to running          │
│                                                              │
│  For each running request r:                                 │
│    2. can_append_slot(r)?                                    │
│       ├─ YES → append_slot(r), schedule for decode          │
│       └─ NO  → preempt(r):                                  │
│           a) swap_out(r) if CPU space available             │
│           b) recompute (discard blocks, re-prefill later)   │
│                                                              │
│  Execute kernel for all running requests.                    │
│                                                              │
│  For each swapped request s:                                 │
│    3. can_swap_in(s)? → swap_in(s), move to running         │
└──────────────────────────────────────────────────────────────┘
```

### 6.8.2 The waterfall metaphor

Think of the scheduler as managing three queues:

```
WAITING ──allocate──▶ RUNNING ──preempt──▶ SWAPPED
                         ▲                    │
                         └──────swap_in───────┘
```

At each step the scheduler tries to fill RUNNING to capacity (limited by GPU
blocks).  When RUNNING is full it stops admitting from WAITING.  When a RUNNING
request exhausts its block budget mid-generation, it is preempted to SWAPPED
(or discarded if no CPU space remains).

### 6.8.3 `[COMMON TRAP]` — Preemption modes

vLLM supports two preemption modes:

```
RECOMPUTE mode:
  Discard all KV blocks for the preempted request.
  When rescheduled: re-run the entire prefill from token 0.
  Pro: no CPU memory needed.
  Con: wasted GPU compute proportional to sequence length.

SWAP mode:
  Move KV blocks to CPU.
  When rescheduled: copy back to GPU.
  Pro: no recompute.
  Con: PCIe bandwidth cost; CPU memory must be available.
```

For short requests (<256 tokens) recompute is usually faster.  For long
requests with expensive system prompts swap is preferred — and prefix caching
makes recompute cheap anyway (only the non-cached suffix must be recomputed).

---

## 6.9 Memory Efficiency Numbers  `[FOUNDATIONAL]`

The vLLM paper (Kwon et al., 2023) reported the following on an A100-40GB with
LLaMA-13B:

```
System              Memory utilization    Peak throughput
──────────────────────────────────────────────────────────
FasterTransformer         20.4 %            1.00× (baseline)
Orca (continuous batch)   ~40 %             ~2.2×
vLLM (PagedAttention)     96.1 %            ~8.5×
──────────────────────────────────────────────────────────
```

The 96.1 % utilization means less than 4 % of the KV pool is wasted — almost
entirely internal fragmentation of the last (partially filled) block per
sequence.

The throughput gain of 8.5× over FasterTransformer comes from two sources:

1. **Batch size**: higher memory utilization allows 5-8× more concurrent
   requests.
2. **Continuous batching**: all decode steps from all requests are batched
   together at every iteration (same as Orca, Chapter 3).

---

## 6.10 Practical Configuration Knobs  `[FOUNDATIONAL]`

### 6.10.1 `--block-size`

Default: 16.  Options: 8, 16, 32.

```
Smaller block_size (8):
  + Less internal fragmentation per request.
  + Finer CoW granularity (cheaper prefix sharing).
  − More block-table indirection in the kernel.
  − Worse block-transfer bandwidth during swaps.

Larger block_size (32):
  + Better kernel efficiency (larger contiguous tiles).
  + Fewer block-table entries.
  − More fragmentation for short requests.
```

For most workloads 16 is optimal.  Use 32 for very long contexts (≥ 4096).

### 6.10.2 `--gpu-memory-utilization`

Default: 0.90.  Controls what fraction of GPU HBM is reserved for KV blocks
after weights.  Reducing it to 0.85 leaves a buffer for activation memory
during long prefills.  Setting it above 0.95 risks OOM during prefill spikes.

### 6.10.3 `--enable-prefix-caching`

Default: off (as of vLLM 0.4).  Enables the block-hash cache.  Almost always
worth enabling when:
- A fixed system prompt is used.
- RAG contexts are prepended.
- Chat history grows across turns.

Cost: a small hash-computation overhead on block allocation (~negligible).

### 6.10.4 `--swap-space`

Default: 4 GB (CPU DRAM reserved for swapped blocks).  Increase on machines
with ample RAM and many concurrent long requests.

---

## 6.11 Code Listing  `[FOUNDATIONAL]`

The following self-contained Python program simulates the PagedAttention block
manager with prefix caching, CoW, and GPU↔CPU swapping.  It requires only
NumPy.

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
# Run:
#   python paged_attention_demo.py

from __future__ import annotations
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Physical block ────────────────────────────────────────────────────────────

@dataclass
class PhysicalBlock:
    block_id:           int
    device:             str   # "gpu" or "cpu"
    ref_count:          int   = 0
    block_hash:         Optional[int] = None
    num_hashed_tokens:  int   = 0

    def __repr__(self):
        return (f"Block(id={self.block_id}, dev={self.device}, "
                f"ref={self.ref_count}, hash={self.block_hash})")


# ── Block allocator ───────────────────────────────────────────────────────────

class BlockAllocatorError(Exception):
    pass


class BlockAllocator:
    """Pool of physical blocks on a single device."""

    def __init__(self, device: str, num_blocks: int, block_size: int):
        self.device     = device
        self.block_size = block_size
        self._free: List[PhysicalBlock] = [
            PhysicalBlock(i, device) for i in range(num_blocks)
        ]
        # prefix cache: hash → block (LRU order via OrderedDict)
        self._cache: OrderedDict[int, PhysicalBlock] = OrderedDict()

    # ── basic alloc / free ────────────────────────────────────────────────

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalBlock:
        """Allocate one block; evict LRU cached block if pool empty."""
        if not self._free:
            self._evict_lru()
        if not self._free:
            raise BlockAllocatorError(f"No free blocks on {self.device}")
        blk = self._free.pop()
        blk.ref_count          = 1
        blk.block_hash         = block_hash
        blk.num_hashed_tokens  = num_hashed_tokens
        if block_hash is not None:
            self._cache[block_hash] = blk
            self._cache.move_to_end(block_hash)   # MRU end
        return blk

    def free(self, blk: PhysicalBlock) -> None:
        blk.ref_count -= 1
        if blk.ref_count == 0:
            if blk.block_hash in self._cache:
                # Keep in cache (speculatively).  Only truly freed on eviction.
                pass
            else:
                self._free.append(blk)

    # ── prefix cache lookup ───────────────────────────────────────────────

    def get_cached(self, block_hash: int) -> Optional[PhysicalBlock]:
        if block_hash in self._cache:
            blk = self._cache[block_hash]
            blk.ref_count += 1
            self._cache.move_to_end(block_hash)
            return blk
        return None

    def _evict_lru(self) -> None:
        """Evict the least-recently-used cached block."""
        for h, blk in self._cache.items():
            if blk.ref_count == 0:
                del self._cache[h]
                blk.block_hash = None
                self._free.append(blk)
                return
        raise BlockAllocatorError("All cached blocks are still referenced")

    @property
    def num_free_blocks(self) -> int:
        # Blocks that are in free list OR in cache with ref_count == 0
        cached_free = sum(1 for b in self._cache.values() if b.ref_count == 0)
        return len(self._free) + cached_free

    def __repr__(self):
        return (f"BlockAllocator(device={self.device}, "
                f"free={len(self._free)}, cached={len(self._cache)})")


# ── Block table ───────────────────────────────────────────────────────────────

class BlockTable:
    """Ordered list of physical blocks for one sequence."""

    def __init__(self, block_size: int, allocator: BlockAllocator):
        self._blocks:    List[PhysicalBlock] = []
        self.block_size: int                 = block_size
        self._allocator: BlockAllocator      = allocator
        self._num_tokens: int                = 0

    # ── prefill ───────────────────────────────────────────────────────────

    def allocate(self, token_ids: List[int], prev_hash: int = 0) -> None:
        """Allocate blocks for a full token list, using prefix cache."""
        full_blocks = len(token_ids) // self.block_size
        remainder   = len(token_ids) %  self.block_size

        for b_idx in range(full_blocks):
            chunk = token_ids[b_idx * self.block_size :
                              (b_idx + 1) * self.block_size]
            h = _block_hash(chunk, prev_hash)
            cached = self._allocator.get_cached(h)
            if cached is not None:
                self._blocks.append(cached)
                print(f"  [prefix-cache HIT] block {b_idx}, hash={h:#010x}")
            else:
                blk = self._allocator.allocate(
                    block_hash=h, num_hashed_tokens=(b_idx + 1) * self.block_size
                )
                self._blocks.append(blk)
                print(f"  [prefix-cache MISS] block {b_idx} → phys_{blk.block_id}")
            prev_hash = h

        if remainder:
            blk = self._allocator.allocate()
            self._blocks.append(blk)
            print(f"  [partial block] block {full_blocks} → phys_{blk.block_id}")

        self._num_tokens = len(token_ids)

    # ── decode: append one token ──────────────────────────────────────────

    def append_slot(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Reserve a slot for the next token.
        Returns (None, None)            — slot found in last block, no CoW
                (old_id, None)          — new block allocated, no CoW
                (old_id, new_id)        — CoW performed
        """
        if not self._blocks:
            raise RuntimeError("No blocks allocated; call allocate() first")

        last = self._blocks[-1]
        tokens_in_last = self._num_tokens % self.block_size
        if tokens_in_last == 0 and self._num_tokens > 0:
            tokens_in_last = self.block_size  # last block is full

        if tokens_in_last < self.block_size:
            # Room in last block.  Check CoW.
            if last.ref_count > 1:
                return self._cow(last)
            return (None, None)
        else:
            # Last block full: allocate new one.
            new_blk = self._allocator.allocate()
            self._blocks.append(new_blk)
            self._num_tokens += 1
            return (new_blk.block_id, None)

    def _cow(self, blk: PhysicalBlock) -> Tuple[int, int]:
        """Copy-on-write: replace shared blk with a private copy."""
        new_blk = self._allocator.allocate()
        new_blk.ref_count = 1
        self._allocator.free(blk)              # decrement ref on old block
        self._blocks[-1] = new_blk
        self._num_tokens += 1
        print(f"  [CoW] phys_{blk.block_id} → phys_{new_blk.block_id}")
        return (blk.block_id, new_blk.block_id)

    def fork(self) -> "BlockTable":
        """Create a child block table sharing all current blocks (for beam search)."""
        child = BlockTable(self.block_size, self._allocator)
        child._blocks     = list(self._blocks)
        child._num_tokens = self._num_tokens
        for blk in child._blocks:
            blk.ref_count += 1
        return child

    @property
    def physical_block_ids(self) -> List[int]:
        return [b.block_id for b in self._blocks]

    def free_all(self) -> None:
        for blk in self._blocks:
            self._allocator.free(blk)
        self._blocks.clear()
        self._num_tokens = 0

    def __repr__(self):
        ids = self.physical_block_ids
        return f"BlockTable(tokens={self._num_tokens}, blocks={ids})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _block_hash(token_ids: List[int], prev_hash: int) -> int:
    raw = str(prev_hash) + str(token_ids)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Basic allocation — three requests
# ═══════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE  = 4
NUM_BLOCKS  = 12

print("=" * 60)
print("SECTION 1: Basic Allocation (block_size=4, pool=12 blocks)")
print("=" * 60)

gpu_alloc = BlockAllocator("gpu", NUM_BLOCKS, BLOCK_SIZE)
print(f"\nInitial state: {gpu_alloc.num_free_blocks} free blocks\n")

# r1: 7 tokens
print("r1 prefill (7 tokens):")
bt_r1 = BlockTable(BLOCK_SIZE, gpu_alloc)
bt_r1.allocate(list(range(7)))
print(f"  {bt_r1}")

# r2: 5 tokens
print("\nr2 prefill (5 tokens):")
bt_r2 = BlockTable(BLOCK_SIZE, gpu_alloc)
bt_r2.allocate(list(range(5)))
print(f"  {bt_r2}")

# Decode: r1 generates 5 more tokens
print("\nr1 decode (5 tokens):")
for step in range(5):
    result = bt_r1.append_slot()
    if result != (None, None):
        print(f"  step {step}: new/cow block event {result}")
print(f"  {bt_r1}")

# r2 completes
print("\nr2 completes (free blocks):")
r2_blocks_before = bt_r2.physical_block_ids[:]
bt_r2.free_all()
print(f"  Freed blocks: {r2_blocks_before}")
print(f"  Free pool size: {gpu_alloc.num_free_blocks}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Copy-on-Write (beam search, B=3)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2: Copy-on-Write — Beam Search (B=3)")
print("=" * 60)

gpu_alloc2 = BlockAllocator("gpu", 20, BLOCK_SIZE)

print("\nShared prefill (12 tokens → 3 full blocks):")
bt_parent = BlockTable(BLOCK_SIZE, gpu_alloc2)
bt_parent.allocate(list(range(12)))
print(f"  Parent: {bt_parent}")

print("\nForking 3 beams:")
beams = [bt_parent.fork() for _ in range(3)]
for i, b in enumerate(beams):
    print(f"  Beam {i}: {b}")

print(f"\nRef counts on shared blocks: "
      f"{[blk.ref_count for blk in beams[0]._blocks]}")

print("\nDecode step 1 (each beam extends):")
for i, beam in enumerate(beams):
    result = beam.append_slot()
    print(f"  Beam {i}: append_slot → {result}")
    print(f"    block table now: {beam.physical_block_ids}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Prefix Caching
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3: Prefix Caching")
print("=" * 60)

gpu_alloc3 = BlockAllocator("gpu", 30, BLOCK_SIZE)

# Shared system prompt: 12 tokens
SYSTEM_PROMPT = list(range(100, 112))  # token IDs 100-111

print(f"\nRequest 1 (cold cache, system_prompt + 4 user tokens):")
tokens_r1 = SYSTEM_PROMPT + [200, 201, 202, 203]
bt_req1 = BlockTable(BLOCK_SIZE, gpu_alloc3)
bt_req1.allocate(tokens_r1)
print(f"  {bt_req1}")

print(f"\nRequest 2 (same system_prompt + 4 different user tokens):")
tokens_r2 = SYSTEM_PROMPT + [300, 301, 302, 303]
bt_req2 = BlockTable(BLOCK_SIZE, gpu_alloc3)
bt_req2.allocate(tokens_r2)
print(f"  {bt_req2}")

# The first 3 blocks (system prompt) should be cache hits
shared = sum(1 for b1, b2 in zip(bt_req1._blocks, bt_req2._blocks)
             if b1.block_id == b2.block_id)
print(f"\n  Shared physical blocks between req1 and req2: {shared}")
print(f"  Free blocks remaining: {gpu_alloc3.num_free_blocks}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: GPU → CPU swap simulation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 4: GPU ↔ CPU Swap")
print("=" * 60)

GPU_BLOCKS = 8
CPU_BLOCKS = 16

gpu_alloc4 = BlockAllocator("gpu", GPU_BLOCKS, BLOCK_SIZE)
cpu_alloc4 = BlockAllocator("cpu", CPU_BLOCKS, BLOCK_SIZE)

def swap_out(bt: BlockTable,
             gpu_alloc: BlockAllocator,
             cpu_alloc: BlockAllocator) -> Dict[int, int]:
    """Move all blocks from GPU to CPU. Returns gpu→cpu id mapping."""
    mapping = {}
    for i, gpu_blk in enumerate(bt._blocks):
        cpu_blk = cpu_alloc.allocate()
        print(f"  copy GPU_{gpu_blk.block_id} → CPU_{cpu_blk.block_id}")
        mapping[gpu_blk.block_id] = cpu_blk.block_id
        gpu_alloc.free(gpu_blk)
        bt._blocks[i] = cpu_blk
    return mapping

def swap_in(bt: BlockTable,
            gpu_alloc: BlockAllocator,
            cpu_alloc: BlockAllocator) -> Dict[int, int]:
    """Move all blocks from CPU back to GPU."""
    mapping = {}
    for i, cpu_blk in enumerate(bt._blocks):
        gpu_blk = gpu_alloc.allocate()
        print(f"  copy CPU_{cpu_blk.block_id} → GPU_{gpu_blk.block_id}")
        mapping[cpu_blk.block_id] = gpu_blk.block_id
        cpu_alloc.free(cpu_blk)
        bt._blocks[i] = gpu_blk
    return mapping

print(f"\nAllocate two sequences on GPU (each 8 tokens = 2 blocks):")
bt_a = BlockTable(BLOCK_SIZE, gpu_alloc4)
bt_a.allocate(list(range(8)))
bt_b = BlockTable(BLOCK_SIZE, gpu_alloc4)
bt_b.allocate(list(range(8, 16)))
print(f"  seq_a: {bt_a}")
print(f"  seq_b: {bt_b}")
print(f"  GPU free blocks: {gpu_alloc4.num_free_blocks}")

print(f"\nGPU full — swap out seq_b:")
mapping_out = swap_out(bt_b, gpu_alloc4, cpu_alloc4)
print(f"  seq_b now on CPU: {bt_b}")
print(f"  GPU free blocks: {gpu_alloc4.num_free_blocks}")

print(f"\nAllocate new request on GPU (now possible):")
bt_c = BlockTable(BLOCK_SIZE, gpu_alloc4)
bt_c.allocate(list(range(16, 24)))
print(f"  seq_c: {bt_c}")

print(f"\nseq_c done, seq_b resumes — swap in seq_b:")
bt_c.free_all()
mapping_in = swap_in(bt_b, gpu_alloc4, cpu_alloc4)
print(f"  seq_b back on GPU: {bt_b}")
print(f"  GPU free blocks: {gpu_alloc4.num_free_blocks}")

print("\nDone.")
```

### 6.11.2 C++ companion

See `code/chapter_06/paged_attention_demo.cpp` for a C++ implementation of the
same block manager, demonstrating:

- `PhysicalBlock` / `BlockAllocator` classes with free-list management
- `BlockTable::append_slot()` with copy-on-write
- Beam-search fork / CoW trace
- Prefix-cache hash chain

---

## 6.12 Chapter Summary

| Concept | Key formula / fact |
|---------|-------------------|
| Naïve KV waste | 60–80 % in typical chat workloads |
| PagedAttention block size | 16 tokens default; 2 MB per block (LLaMA 3 8B) |
| Block table lookup | `physical_id = block_table[logical_id // block_size]` |
| CoW trigger | `ref_count > 1` on the last block when appending |
| Prefix cache hash | `hash(token_ids, prev_block_hash)` — chain hash |
| vLLM memory utilization | 96.1 % vs ~20 % for FasterTransformer |
| Throughput gain | ~8.5× over FasterTransformer (A100, LLaMA-13B) |
| Swap cost | ~62.5 µs per block over PCIe 4.0 |

### Why this matters for what follows

- **Chapter 7** (Continuous Batching Scheduler) uses the block manager's
  `can_allocate` / `can_append_slot` APIs as its core decision predicates.
- **Chapter 9** (Speculative Decoding) needs CoW to cheaply fork the KV state
  for draft-model rollouts.
- **Chapter 13** (Disaggregated Prefill) moves the prefill computation to a
  separate machine; PagedAttention's block-level granularity makes the resulting
  KV transfer tractable.
- **Chapter 15** (Long-context Systems) relies on prefix caching to make
  repeated long-context queries affordable.

---

## 6.13 Further Reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving
  with PagedAttention," SOSP 2023.  *(The primary source.)*
- vLLM source: `vllm/core/block_manager.py`, `vllm/core/scheduler.py`.
- Orca: Yu et al., "Orca: A Distributed Serving System for Transformer-Based
  Generative Models," OSDI 2022.
- Sheng et al., "FlexGen: High-Throughput Generative Inference of Large
  Language Models with a Single GPU," ICML 2023.  *(CPU offloading.)*

---

*End of Chapter 6.*


---

## Chapter Summary

- **The KV cache fragmentation problem**: naive per-sequence pre-allocation wastes 60–80 % of GPU memory on average because sequences have unpredictable lengths.
- **Virtual memory analogy**: PagedAttention maps logical KV blocks to non-contiguous physical blocks, exactly as a virtual memory system maps virtual pages to physical frames.
- **Block granularity**: vLLM defaults to 16-token blocks; each block holds K and V tensors for all heads for those 16 tokens, contiguous within the block.
- **Block table**: the sequence-level data structure that holds the logical→physical mapping, updated every time a new block is allocated.
- **Copy-on-write**: shared prefix blocks are reference-counted; a write triggers a CoW copy, enabling beam search without duplicating the entire KV cache.
- **Prefix caching**: blocks whose token sequence matches a previous request can be reused unchanged, reducing prefill FLOPs by up to 100 % for cached prefixes.
- **Swapping to CPU**: when GPU blocks are exhausted, vLLM can swap lower-priority sequences to CPU DRAM (62.5 µs/block over PCIe 4.0) and swap them back on resumption.
- **Throughput impact**: PagedAttention achieves ~8.5× throughput improvement over FasterTransformer on A100 (LLaMA-13B).

---

## Self-Check Questions

1. A naive KV cache allocates `max_seq_len × d_model × 2 × 2` bytes per sequence upfront (2 for K and V, 2 for FP16). For max_seq_len = 4 096, d_model = 4 096, and a batch of 32, compute the total allocation. How much is wasted if average sequence length is 512 tokens? *(Section 6.1)*

2. vLLM uses 16-token blocks with 32 KV heads and d_k = 128 in FP16. Compute the byte size of one physical block. *(Section 6.3)*

3. Beam search with width 4 forks a sequence of 200 tokens into 4 candidates. Without CoW, how many bytes of KV cache are needed? With CoW (assuming the 200-token prefix is shared), how many bytes? (Use the same block parameters as above.) *(Section 6.5)*

4. Prefix caching stores the KV blocks for a "system prompt" of 512 tokens that 1 000 users share. What is the total KV cache saving (in bytes, using the block parameters above) compared to recomputing the prefix every time? *(Section 6.6)*

5. The block manager calls `can_append_slot(seq)` before each decode step. Describe all the conditions under which this call returns False, and what the scheduler does in each case. *(Section 6.4)*
