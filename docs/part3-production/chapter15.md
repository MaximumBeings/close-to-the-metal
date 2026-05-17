# Chapter 15: Multi-GPU Serving and Tensor Parallelism

> **"Adding a second GPU is not adding a second engine — it is splitting one engine across two chassis and paying a tax every time the halves need to talk."**

---

## What This Chapter Covers

Single-GPU throughput hits a hard ceiling: the bandwidth wall of one HBM die and the capacity of one block of SRAM. For the largest models — 70 B parameters and above — that ceiling is reached before the model even fits in memory. Multi-GPU serving is the industry's answer, but it comes with latency costs, configuration complexity, and failure modes that a practitioner must understand before pulling the trigger.

This chapter covers tensor parallelism (TP) end-to-end: how weights are sharded, what the AllReduce collective costs, when NVLink makes TP nearly free versus when PCIe makes it painful, how the KV cache is divided across GPUs, and how vLLM's `tensor_parallel_size` knob and llama.cpp's `--tensor-split` flag map onto these mechanisms. It closes with a decision framework: when to add GPUs, and when to scale out with more single-GPU replicas instead.

---

## 15.1 Why One GPU Is Not Enough

### The capacity problem

A BF16 model parameter costs 2 bytes. Llama-3-70B has 70 billion of them:

```
70 × 10^9 × 2 = 140 GB
```

An A100-80 has 80 GB of HBM. The model does not fit. An H100 with 80 GB HBM also does not fit. Even an H200 with 141 GB barely fits — and that leaves almost nothing for KV cache.

**[FOUNDATIONAL]** The only options for serving a 70 B model are: (a) quantise aggressively enough to halve the weight footprint, (b) spread the model across multiple GPUs, or (c) use CPU offload with its severe throughput penalty. Most production deployments choose (b).

### The bandwidth wall

Even when a model fits — say Llama-3-8B on a single A100-80 — the decode phase is memory-bandwidth-bound. Every decode step must load the entire weight matrix from HBM to compute one forward pass. For 8 B parameters at BF16:

```
Weight bytes per step = 8 × 10^9 × 2 = 16 GB
A100-80 bandwidth    = 2 TB/s

Decode throughput ceiling = 2 × 10^12 / 16 × 10^9 = 125 tok/s (single GPU)
```

Two GPUs each hold half the weights. Each GPU loads 8 GB per step instead of 16 GB, so the theoretical ceiling doubles — but only if the two halves can synchronize cheaply. That cost is the AllReduce.

---

## 15.2 Tensor Parallelism: Splitting the Weight Matrix

Tensor parallelism divides individual weight matrices across GPUs within a single model layer. Unlike pipeline parallelism (which assigns different layers to different GPUs), TP keeps all GPUs busy on every token of every layer.

**Figure 15.1 — Tensor Parallel Column-then-Row Split across 4 GPUs**

<img src="data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgNzAwIDM2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICAgIHN0eWxlPSJtYXgtd2lkdGg6NzIwcHg7Zm9udC1mYW1pbHk6R2VvcmdpYSxzZXJpZjtmb250LXNpemU6MTJweDtkaXNwbGF5OmJsb2NrIj4KICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFycjE1IiBtYXJrZXJXaWR0aD0iOCIgbWFya2VySGVpZ2h0PSI4IiByZWZYPSI0IiByZWZZPSI0IiBvcmllbnQ9ImF1dG8iPgogICAgICA8cGF0aCBkPSJNMSwxIEw3LDQgTDEsNyB6IiBmaWxsPSIjMzc0MTUxIi8+CiAgICA8L21hcmtlcj4KICAgIDxtYXJrZXIgaWQ9ImFycjE1ciIgbWFya2VyV2lkdGg9IjgiIG1hcmtlckhlaWdodD0iOCIgcmVmWD0iNCIgcmVmWT0iNCIgb3JpZW50PSJhdXRvIj4KICAgICAgPHBhdGggZD0iTTcsMSBMMSw0IEw3LDcgeiIgZmlsbD0iIzM3NDE1MSIvPgogICAgPC9tYXJrZXI+CiAgPC9kZWZzPgogIDxyZWN0IHdpZHRoPSI3MDAiIGhlaWdodD0iMzYwIiBmaWxsPSIjZjlmYWZiIiByeD0iNiIgc3Ryb2tlPSIjZTVlN2ViIi8+CgogIDwhLS0gVGl0bGUgLS0+CiAgPHRleHQgeD0iMzUwIiB5PSIyNSIgZmlsbD0iIzExMTgyNyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxNCIgZm9udC13ZWlnaHQ9ImJvbGQiPlRlbnNvciBQYXJhbGxlbGlzbSAoVFA9NCkg4oCUIENvbHVtbiArIFJvdyBTcGxpdDwvdGV4dD4KCiAgPCEtLSBJbnB1dCBicm9hZGNhc3QgLS0+CiAgPHJlY3QgeD0iMjgwIiB5PSI0MCIgd2lkdGg9IjE0MCIgaGVpZ2h0PSIzNCIgcng9IjUiIGZpbGw9IiNkYmVhZmUiIHN0cm9rZT0iIzI1NjNlYiIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgPHRleHQgeD0iMzUwIiB5PSI2MyIgZmlsbD0iIzFlNDBhZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMiIgZm9udC13ZWlnaHQ9ImJvbGQiPklucHV0IHggIFtCLCBkXTwvdGV4dD4KICA8dGV4dCB4PSIzNTAiIHk9IjgyIiBmaWxsPSIjNmI3MjgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIj5icm9hZGNhc3QgdG8gYWxsIDQgR1BVczwvdGV4dD4KICA8bGluZSB4MT0iMzUwIiB5MT0iNzQiIHgyPSIzNTAiIHkyPSI5MCIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEiIHN0cm9rZS1kYXNoYXJyYXk9IjMsMiIvPgoKICA8IS0tIEdQVSBib3hlczogNCBHUFVzIHNpZGUgYnkgc2lkZSAtLT4KICA8IS0tIEdQVSAwIC0tPgogIDxyZWN0IHg9IjMwIiAgeT0iMTAwIiB3aWR0aD0iMTM4IiBoZWlnaHQ9IjE0MCIgcng9IjUiIGZpbGw9IiNlZmY2ZmYiIHN0cm9rZT0iIzNiODJmNiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KICA8dGV4dCB4PSI5OSIgIHk9IjExOCIgZmlsbD0iIzFlNDBhZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9ImJvbGQiPkdQVSAwPC90ZXh0PgogIDxyZWN0IHg9IjQ1IiAgeT0iMTI1IiB3aWR0aD0iMTA4IiBoZWlnaHQ9IjM0IiByeD0iMyIgZmlsbD0iI2JmZGJmZSIgc3Ryb2tlPSIjMjU2M2ViIi8+CiAgPHRleHQgeD0iOTkiICB5PSIxNDAiIGZpbGw9IiMxZTNhOGEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiIGZvbnQtd2VpZ2h0PSJib2xkIj5X4oKBWzosMDpkLzRdPC90ZXh0PgogIDx0ZXh0IHg9Ijk5IiAgeT0iMTU0IiBmaWxsPSIjNmI3MjgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPmNvbHMgMOKApmQvNDwvdGV4dD4KICA8cmVjdCB4PSI0NSIgIHk9IjE2OCIgd2lkdGg9IjEwOCIgaGVpZ2h0PSIzNCIgcng9IjMiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIvPgogIDx0ZXh0IHg9Ijk5IiAgeT0iMTgzIiBmaWxsPSIjMTY2NTM0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+V+KCglswOmQvNCw6XTwvdGV4dD4KICA8dGV4dCB4PSI5OSIgIHk9IjE5NyIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI5Ij5yb3dzIDDigKZkLzQ8L3RleHQ+CiAgPHRleHQgeD0iOTkiICB5PSIyMjQiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+cGFydGlhbCBvdXRwdXQgeeKCgDwvdGV4dD4KCiAgPCEtLSBHUFUgMSAtLT4KICA8cmVjdCB4PSIxODgiIHk9IjEwMCIgd2lkdGg9IjEzOCIgaGVpZ2h0PSIxNDAiIHJ4PSI1IiBmaWxsPSIjZWZmNmZmIiBzdHJva2U9IiMzYjgyZjYiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPHRleHQgeD0iMjU3IiB5PSIxMTgiIGZpbGw9IiMxZTQwYWYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSJib2xkIj5HUFUgMTwvdGV4dD4KICA8cmVjdCB4PSIyMDMiIHk9IjEyNSIgd2lkdGg9IjEwOCIgaGVpZ2h0PSIzNCIgcng9IjMiIGZpbGw9IiNiZmRiZmUiIHN0cm9rZT0iIzI1NjNlYiIvPgogIDx0ZXh0IHg9IjI1NyIgeT0iMTQwIiBmaWxsPSIjMWUzYThhIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+V+KCgVs6LGQvNDpkLzJdPC90ZXh0PgogIDx0ZXh0IHg9IjI1NyIgeT0iMTU0IiBmaWxsPSIjNmI3MjgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPmNvbHMgZC804oCmZC8yPC90ZXh0PgogIDxyZWN0IHg9IjIwMyIgeT0iMTY4IiB3aWR0aD0iMTA4IiBoZWlnaHQ9IjM0IiByeD0iMyIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIi8+CiAgPHRleHQgeD0iMjU3IiB5PSIxODMiIGZpbGw9IiMxNjY1MzQiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiIGZvbnQtd2VpZ2h0PSJib2xkIj5X4oKCW2QvNDpkLzIsOl08L3RleHQ+CiAgPHRleHQgeD0iMjU3IiB5PSIxOTciIGZpbGw9IiM2YjcyODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+cm93cyBkLzTigKZkLzI8L3RleHQ+CiAgPHRleHQgeD0iMjU3IiB5PSIyMjQiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+cGFydGlhbCBvdXRwdXQgeeKCgTwvdGV4dD4KCiAgPCEtLSBHUFUgMiAtLT4KICA8cmVjdCB4PSIzNDYiIHk9IjEwMCIgd2lkdGg9IjEzOCIgaGVpZ2h0PSIxNDAiIHJ4PSI1IiBmaWxsPSIjZWZmNmZmIiBzdHJva2U9IiMzYjgyZjYiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPHRleHQgeD0iNDE1IiB5PSIxMTgiIGZpbGw9IiMxZTQwYWYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSJib2xkIj5HUFUgMjwvdGV4dD4KICA8cmVjdCB4PSIzNjEiIHk9IjEyNSIgd2lkdGg9IjEwOCIgaGVpZ2h0PSIzNCIgcng9IjMiIGZpbGw9IiNiZmRiZmUiIHN0cm9rZT0iIzI1NjNlYiIvPgogIDx0ZXh0IHg9IjQxNSIgeT0iMTQwIiBmaWxsPSIjMWUzYThhIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+V+KCgVs6LGQvMjozZC80XTwvdGV4dD4KICA8dGV4dCB4PSI0MTUiIHk9IjE1NCIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI5Ij5jb2xzIGQvMuKApjNkLzQ8L3RleHQ+CiAgPHJlY3QgeD0iMzYxIiB5PSIxNjgiIHdpZHRoPSIxMDgiIGhlaWdodD0iMzQiIHJ4PSIzIiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiLz4KICA8dGV4dCB4PSI0MTUiIHk9IjE4MyIgZmlsbD0iIzE2NjUzNCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMCIgZm9udC13ZWlnaHQ9ImJvbGQiPlfigoJbM2QvNDpkLDpdPC90ZXh0PgogIDx0ZXh0IHg9IjQxNSIgeT0iMTk3IiBmaWxsPSIjNmI3MjgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjkiPnJvd3MgM2QvNOKApmQ8L3RleHQ+CiAgPHRleHQgeD0iNDE1IiB5PSIyMjQiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+cGFydGlhbCBvdXRwdXQgeeKCgjwvdGV4dD4KCiAgPCEtLSBHUFUgMyAtLT4KICA8cmVjdCB4PSI1MDQiIHk9IjEwMCIgd2lkdGg9IjE2OCIgaGVpZ2h0PSIxNDAiIHJ4PSI1IiBmaWxsPSIjZWZmNmZmIiBzdHJva2U9IiMzYjgyZjYiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPHRleHQgeD0iNTg4IiB5PSIxMTgiIGZpbGw9IiMxZTQwYWYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSJib2xkIj5HUFUgMzwvdGV4dD4KICA8cmVjdCB4PSI1MTkiIHk9IjEyNSIgd2lkdGg9IjEzOCIgaGVpZ2h0PSIzNCIgcng9IjMiIGZpbGw9IiNiZmRiZmUiIHN0cm9rZT0iIzI1NjNlYiIvPgogIDx0ZXh0IHg9IjU4OCIgeT0iMTQwIiBmaWxsPSIjMWUzYThhIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+V+KCgVs6LDNkLzQ6ZF08L3RleHQ+CiAgPHRleHQgeD0iNTg4IiB5PSIxNTQiIGZpbGw9IiM2YjcyODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+Y29scyAzZC804oCmZDwvdGV4dD4KICA8cmVjdCB4PSI1MTkiIHk9IjE2OCIgd2lkdGg9IjEzOCIgaGVpZ2h0PSIzNCIgcng9IjMiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIvPgogIDx0ZXh0IHg9IjU4OCIgeT0iMTgzIiBmaWxsPSIjMTY2NTM0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+V+KCgltkLzI6M2QvNCw6XTwvdGV4dD4KICA8dGV4dCB4PSI1ODgiIHk9IjE5NyIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSI5Ij5yb3dzIGQvMuKApjNkLzQ8L3RleHQ+CiAgPHRleHQgeD0iNTg4IiB5PSIyMjQiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iOSI+cGFydGlhbCBvdXRwdXQgeeKCgzwvdGV4dD4KCiAgPCEtLSBBcnJvd3MgaW5wdXQg4oaSIEdQVXMgLS0+CiAgPGxpbmUgeDE9IjI2MCIgeTE9IjkwIiB4Mj0iOTkiICB5Mj0iMTAwIiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMSIgc3Ryb2tlLWRhc2hhcnJheT0iMywyIiBtYXJrZXItZW5kPSJ1cmwoI2FycjE1KSIvPgogIDxsaW5lIHgxPSIzMTAiIHkxPSI5MCIgeDI9IjI1NyIgeTI9IjEwMCIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEiIHN0cm9rZS1kYXNoYXJyYXk9IjMsMiIgbWFya2VyLWVuZD0idXJsKCNhcnIxNSkiLz4KICA8bGluZSB4MT0iMzkwIiB5MT0iOTAiIHgyPSI0MTUiIHkyPSIxMDAiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSIxIiBzdHJva2UtZGFzaGFycmF5PSIzLDIiIG1hcmtlci1lbmQ9InVybCgjYXJyMTUpIi8+CiAgPGxpbmUgeDE9IjQ0MCIgeTE9IjkwIiB4Mj0iNTg4IiB5Mj0iMTAwIiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMSIgc3Ryb2tlLWRhc2hhcnJheT0iMywyIiBtYXJrZXItZW5kPSJ1cmwoI2FycjE1KSIvPgoKICA8IS0tIEFsbFJlZHVjZSBiYXIgLS0+CiAgPHJlY3QgeD0iMzAiIHk9IjI2MiIgd2lkdGg9IjY0MiIgaGVpZ2h0PSIzNCIgcng9IjUiIGZpbGw9IiNmZWYzYzciIHN0cm9rZT0iI2Q5NzcwNiIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgPHRleHQgeD0iMzUwIiB5PSIyODMiIGZpbGw9IiM5MjQwMGUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtd2VpZ2h0PSJib2xkIj5BbGxSZWR1Y2U6IHkgPSB54oKAICsgeeKCgSArIHnigoIgKyB54oKDICAoTlZMaW5rIHJpbmcpPC90ZXh0PgoKICA8IS0tIEFycm93cyBHUFUg4oaSIEFsbFJlZHVjZSAtLT4KICA8bGluZSB4MT0iOTkiICB5MT0iMjQwIiB4Mj0iOTkiICB5Mj0iMjYyIiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycjE1KSIvPgogIDxsaW5lIHgxPSIyNTciIHkxPSIyNDAiIHgyPSIyNTciIHkyPSIyNjIiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyMTUpIi8+CiAgPGxpbmUgeDE9IjQxNSIgeTE9IjI0MCIgeDI9IjQxNSIgeTI9IjI2MiIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIxNSkiLz4KICA8bGluZSB4MT0iNTg4IiB5MT0iMjQwIiB4Mj0iNTg4IiB5Mj0iMjYyIiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycjE1KSIvPgoKICA8IS0tIE91dHB1dCAtLT4KICA8bGluZSB4MT0iMzUwIiB5MT0iMjk2IiB4Mj0iMzUwIiB5Mj0iMzE2IiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycjE1KSIvPgogIDxyZWN0IHg9IjI2NSIgeT0iMzE2IiB3aWR0aD0iMTcwIiBoZWlnaHQ9IjMyIiByeD0iNSIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjIiLz4KICA8dGV4dCB4PSIzNTAiIHk9IjMzNyIgZmlsbD0iIzE2NjUzNCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMiIgZm9udC13ZWlnaHQ9ImJvbGQiPk91dHB1dCB5ICBbQiwgZF0gIOKckzwvdGV4dD4KCiAgPCEtLSBMZWdlbmQgYm90dG9tIHJpZ2h0IC0tPgogIDxyZWN0IHg9IjMwIiB5PSIzMDgiIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgcng9IjIiIGZpbGw9IiNiZmRiZmUiIHN0cm9rZT0iIzI1NjNlYiIvPgogIDx0ZXh0IHg9IjUwIiB5PSIzMjAiIGZpbGw9IiMzNzQxNTEiIGZvbnQtc2l6ZT0iOSI+Q29sdW1uLXBhcmFsbGVsIFfigoE8L3RleHQ+CiAgPHJlY3QgeD0iMTYwIiB5PSIzMDgiIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgcng9IjIiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIvPgogIDx0ZXh0IHg9IjE4MCIgeT0iMzIwIiBmaWxsPSIjMzc0MTUxIiBmb250LXNpemU9IjkiPlJvdy1wYXJhbGxlbCBX4oKCPC90ZXh0Pgo8L3N2Zz4=" style="max-width:720px;width:100%;display:block;margin:1rem 0" alt="diagram"/>

### Column-then-row splitting (Megatron-LM style)

The two large matrix multiplications in a transformer block are the MLP projections:

```
FFN: x → up_proj → act → down_proj → output
```

For a 4096-dimensional hidden state and 16384-dimensional FFN intermediate:

```
up_proj   shape: [4096, 16384]   (16 384 columns)
down_proj shape: [16384, 4096]   (4 096 columns)
```

With TP=2, each GPU holds:

```
GPU 0: up_proj[:, :8192]     down_proj[:8192, :]
GPU 1: up_proj[:, 8192:]     down_proj[8192:, :]
```

The forward pass becomes:

```
Step 1 (parallel, no communication):
  GPU 0: partial_0 = x @ up_proj_0   shape [batch, 8192]
  GPU 1: partial_1 = x @ up_proj_1   shape [batch, 8192]

Step 2 (parallel, no communication):
  GPU 0: act_0 = gelu(partial_0)
  GPU 1: act_1 = gelu(partial_1)

Step 3 (parallel, no communication):
  GPU 0: out_0 = act_0 @ down_proj_0   shape [batch, 4096]
  GPU 1: out_1 = act_1 @ down_proj_1   shape [batch, 4096]

Step 4 (communication required):
  AllReduce: output = out_0 + out_1     shape [batch, 4096]
```

**[FOUNDATIONAL]** Each transformer layer requires exactly **two AllReduces** (one after the attention output projection, one after the FFN down-projection). With L layers and two AllReduces per layer, a model forward pass incurs 2L collective synchronization operations.

### Attention head splitting

The Q, K, and V projection matrices are split along the head dimension:

```
Q shape: [hidden, n_heads × head_dim]
```

With TP=2 and 32 heads:

```
GPU 0: Q[:, :16 × 128]   K[:, :16 × 128]   V[:, :16 × 128]
GPU 1: Q[:, 16×128:]     K[:, 16×128:]     V[:, 16×128:]
```

Each GPU computes attention over its local heads. After the output projection (also split), an AllReduce reassembles the full hidden state.

**[DEEP DIVE]** For GQA (Grouped Query Attention) models like Llama-3 (32 attention heads, 8 KV heads), TP shards KV heads too. With TP=2, each GPU has 4 KV heads instead of 8. This directly halves the KV cache footprint per GPU — a useful side effect of TP, not just a cost.

```
KV bytes per token per GPU = 2 × layers × kv_heads_per_gpu × head_dim × dtype_bytes
                           = 2 × 32 × 4 × 128 × 2        (TP=2, Llama-3-8B)
                           = 65,536 bytes per token
```

Compare to single-GPU: 131,072 bytes per token. TP=2 halves KV memory pressure.

---

## 15.3 The AllReduce Collective

### What AllReduce does

AllReduce takes partial results from N GPUs and produces the sum (or other reduction) on all N GPUs simultaneously. The operation used in transformer TP is sum-reduce.

```
Before AllReduce:
  GPU 0: out_0 = [1.2, 0.8, ...]   (partial activations)
  GPU 1: out_1 = [0.3, 1.1, ...]   (partial activations)

After AllReduce (both GPUs see the result):
  GPU 0: out = [1.5, 1.9, ...]
  GPU 1: out = [1.5, 1.9, ...]
```

### Ring-AllReduce algorithm

The standard implementation is ring-allreduce, which avoids a bottleneck master node:

```
Phase 1 — Scatter-reduce (N-1 steps):
  Each GPU sends a chunk to its ring-neighbor and receives a chunk
  from its other neighbor, accumulating partial sums.
  After N-1 steps, each GPU holds the full reduction for one chunk.

Phase 2 — All-gather (N-1 steps):
  Each GPU broadcasts its fully-reduced chunk around the ring.
  After N-1 steps, all GPUs have the complete result.

Total data transmitted per GPU = 2 × (N-1)/N × message_size ≈ 2 × message_size
```

For large N this converges to 2× the message size, making ring-allreduce communication-efficient regardless of GPU count.

### AllReduce message size for transformer TP

Each AllReduce transfers the full hidden-state activation tensor:

```
Message size = batch_tokens × hidden_dim × dtype_bytes
             = 128 × 4096 × 2
             = 1,048,576 bytes  (1 MB for batch=128, hidden=4096)
```

With 2L AllReduces per forward pass, total communication volume per step:

```
Volume = 2 × 2L × batch_tokens × hidden_dim × dtype_bytes
       = 2 × 2 × 32 × 128 × 4096 × 2   (Llama-3-8B, L=32, batch=128)
       = 134,217,728 bytes  ≈ 128 MB
```

---

## 15.4 NVLink vs PCIe: The Interconnect Determines Everything

The AllReduce bandwidth determines the tax. Two interconnects dominate modern GPU servers.

### NVLink (within a node, direct GPU-to-GPU)

| Generation | Bandwidth (bidirectional) | Latency |
|-----------|--------------------------|---------|
| NVLink 3.0 (A100) | 600 GB/s | ~1 µs |
| NVLink 4.0 (H100) | 900 GB/s | ~1 µs |

For 128 MB of AllReduce data on two A100s connected by NVLink:

```
AllReduce time ≈ 2 × 128 MB / 600 GB/s = 0.43 ms
```

A single A100 forward pass for 128-token decode batch takes approximately:

```
Decode latency ≈ weight_bytes / bandwidth = 16 GB / 2 TB/s = 8 ms
```

AllReduce overhead fraction = 0.43 / 8 ≈ **5%** — essentially free.

### PCIe (across nodes or consumer GPUs)

| Configuration | Bandwidth (bidirectional) | Latency |
|--------------|--------------------------|---------|
| PCIe 4.0 ×16 | 32 GB/s | ~5 µs |
| PCIe 5.0 ×16 | 64 GB/s | ~3 µs |

For the same 128 MB on PCIe 4.0:

```
AllReduce time ≈ 2 × 128 MB / 32 GB/s = 8 ms
```

AllReduce overhead fraction = 8 / 8 ≈ **100%** — doubles decode latency.

**[COMMON TRAP]** Consumer workstations connect GPUs via PCIe. A developer who tests TP=2 on two RTX 4090s and sees 50% slower latency than a single 4090 is not making an error — PCIe AllReduce is genuinely that expensive. The same configuration on an A100 server with NVLink shows near-linear throughput scaling. **Never benchmark TP on PCIe hardware and expect the numbers to transfer to NVLink production servers.**

### Worked example: break-even batch size

AllReduce latency is fixed per step (dominated by message size and link speed). Compute latency scales with batch size. There exists a batch size below which PCIe TP is slower than a single GPU:

```
Compute time    = batch_tokens × 2 × params × dtype / peak_flops
AllReduce time  = 2 × 2L × batch_tokens × hidden × dtype / bandwidth

For PCIe:
  AllReduce time ≈ constant_per_MB × 128  [fixed for hidden=4096, L=32]

Break-even: compute_per_token × B = allreduce_overhead
  B_breakeven = allreduce_time × peak_flops / (2 × params × dtype)
```

For two RTX 4090s on PCIe 4.0 serving Llama-3-8B:

```
AllReduce time   ≈ 2 × 128 MB / 32 GB/s = 8 ms
Peak flops       = 165 TFLOPS  
Compute per tok  = 2 × 8 × 10^9 × 2 / (165 × 10^12) = 0.19 ms

Break-even B     = 8 ms / 0.19 ms = ~42 tokens
```

Below 42 tokens per step (typical for low-concurrency decode), the PCIe AllReduce costs more than the compute it is meant to parallelize. **A single 4090 outperforms two 4090s in PCIe TP at low batch sizes.**

---

## 15.5 ASCII Diagram: TP=2 Forward Pass

```
                    Input hidden state (batch × hidden)
                           │
              ┌────────────┴────────────┐
              │ broadcast (or duplicate) │
              └────────────┬────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                                 ▼
    ┌──────────┐                      ┌──────────┐
    │  GPU 0   │                      │  GPU 1   │
    │          │                      │          │
    │ W_Q[:,:H/2]                    │ W_Q[:,H/2:]
    │ W_K[:,:H/2]                    │ W_K[:,H/2:]
    │ W_V[:,:H/2]                    │ W_V[:,H/2:]
    │          │                      │          │
    │ Attention│                      │ Attention│
    │ (heads   │                      │ (heads   │
    │  0..H/2-1)                     │  H/2..H-1)
    │          │                      │          │
    │ partial  │                      │ partial  │
    │ out_0    │                      │ out_1    │
    └────┬─────┘                      └────┬─────┘
         │                                 │
         └──────────────┬──────────────────┘
                        │
                  ┌─────▼─────┐
                  │  AllReduce │  ← NVLink: ~0.4 ms
                  │  (sum)     │    PCIe:   ~8 ms
                  └─────┬─────┘
                        │
              full hidden state restored
                        │
          ┌─────────────┼─────────────┐
          ▼                           ▼
    ┌──────────┐                 ┌──────────┐
    │  GPU 0   │                 │  GPU 1   │
    │ FFN half │                 │ FFN half │
    └────┬─────┘                 └────┬─────┘
         │                            │
         └────────────┬───────────────┘
                      │
                ┌─────▼─────┐
                │  AllReduce │
                └─────┬─────┘
                      │
               layer output
```

Two AllReduces per layer. With L=32 layers: 64 AllReduces per forward pass.

---

## 15.6 Pipeline Parallelism (PP): A Brief Note

Pipeline parallelism assigns different layers to different GPUs rather than splitting each layer. GPU 0 runs layers 1–16, GPU 1 runs layers 17–32. A micro-batch flows through GPU 0, then GPU 1 in sequence.

**Why PP is not usually preferred for inference serving:**

- **Bubble latency:** when the pipeline is filling or draining, some GPUs are idle ("pipeline bubble"). Bubble fraction = (PP - 1) / PP for a single request.
- **Inter-stage bandwidth:** activations must be transferred between stage GPUs at every layer boundary — same interconnect tax as TP, but the message is a full activation tensor rather than a reduction.
- **Variable-length sequences:** padding is required to keep pipeline stages aligned, wasting compute.

TP is the default in vLLM. PP is available but requires care and is typically reserved for extremely large models (>405 B parameters) where TP alone would require too many GPUs per node.

---

## 15.7 vLLM: tensor_parallel_size

### What the knob does

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=4,         # ← this knob
    gpu_memory_utilization=0.90,
)
```

Setting `tensor_parallel_size=N` causes vLLM to:

1. Spawn N worker processes (one per GPU via NCCL).
2. Shard all weight tensors along the head/neuron dimension.
3. Wrap each AllReduce call with an NCCL collective.
4. Divide the KV pool across N GPUs, each holding 1/N of the KV heads (or all KV heads with a min-clamped division for GQA).
5. Route all inference through the N-GPU ring.

### Constraints on tensor_parallel_size

```
Constraint 1: tensor_parallel_size must divide num_attention_heads evenly
  Llama-3-70B has 64 heads → valid TP: 1, 2, 4, 8, 16, 32, 64

Constraint 2: tensor_parallel_size must divide num_kv_heads evenly (GQA)
  Llama-3-70B has 8 KV heads → valid TP: 1, 2, 4, 8

Constraint 3: tensor_parallel_size ≤ physical GPUs on the node (NVLink domain)
  Crossing nodes requires pipeline parallelism or expert parallelism

Combining: Llama-3-70B valid TP on a single 8-GPU DGX: {1, 2, 4, 8}
```

**[COMMON TRAP]** Setting `tensor_parallel_size=8` on a model with 8 KV heads gives each GPU exactly 1 KV head. While legal, this results in KV cache tensors of shape [layers, 1, head_dim] per GPU, with very little spatial locality. Prefix caching hash granularity degrades. In practice, TP=4 often yields better end-to-end throughput than TP=8 for 70 B models when 4 GPUs have enough memory to hold the model.

### Effect on KV pool

```
KV bytes per token (single GPU) = 2 × L × kv_heads × head_dim × dtype
KV bytes per token (TP=N)       = 2 × L × (kv_heads / N) × head_dim × dtype

With Llama-3-70B, L=80, kv_heads=8, head_dim=128, dtype=2:
  TP=1: 2 × 80 × 8 × 128 × 2 = 327,680 B/token
  TP=2: 2 × 80 × 4 × 128 × 2 = 163,840 B/token
  TP=4: 2 × 80 × 2 × 128 × 2 =  81,920 B/token
  TP=8: 2 × 80 × 1 × 128 × 2 =  40,960 B/token
```

This is a direct benefit: more TP → less KV pressure per GPU → larger KV pool → higher concurrency.

### Memory budget at each TP level

For Llama-3-70B on A100-80:

```
TP=1 (single A100-80):
  Weights: 140 GB → does not fit (80 GB HBM)

TP=2 (two A100-80s, 160 GB total):
  Weights per GPU: 70 GB
  Usable HBM:     72 GB (90% of 80)
  KV pool:         0 GB → barely fits, essentially no KV cache

TP=4 (four A100-80s, 320 GB total):
  Weights per GPU: 35 GB
  Usable HBM:      72 GB
  KV pool:         35 GB → 35 × 10^9 / 81,920 ≈ 427,000 tokens

TP=8 (eight A100-80s, 640 GB total):
  Weights per GPU: 17.5 GB
  Usable HBM:      72 GB
  KV pool:         53 GB → 53 × 10^9 / 40,960 ≈ 1,294,000 tokens
```

**[DEEP DIVE]** Practical deployments of Llama-3-70B almost always use TP=4 on A100-80 (the minimum that gives a workable KV pool) or TP=2 on H100/H200 where the higher-bandwidth memory and larger HBM give more room. TP=8 trades AllReduce cost (now 8 GPUs, each step sends to 7 neighbors) for KV capacity — sometimes worth it for RAG workloads with very long contexts.

---

## 15.8 llama.cpp: --tensor-split

llama.cpp implements tensor parallelism via `--tensor-split`, which specifies the fraction of the model to place on each GPU:

```bash
# Equal split across two GPUs
llama-server \
  --model /models/llama-3-70b-q4_k_m.gguf \
  --n-gpu-layers 80 \
  --tensor-split 1,1 \
  --ctx-size 16384 \
  --parallel 4

# Unequal split: GPU 0 (24 GB VRAM) gets 40%, GPU 1 (48 GB) gets 60%
llama-server \
  --model /models/llama-3-70b-q4_k_m.gguf \
  --n-gpu-layers 80 \
  --tensor-split 2,3 \
  --parallel 4
```

The fractions in `--tensor-split` are ratios, not percentages. `1,1` means equal split; `2,3` means 40% / 60%.

### How llama.cpp TP differs from vLLM TP

| Aspect | vLLM | llama.cpp |
|--------|------|-----------|
| Parallelism type | True tensor parallel (Megatron style) | Layer offload split (not full TP) |
| AllReduce | NCCL ring allreduce | Metal / CUDA peer copy |
| Communication | NVLink or PCIe | PCIe (on most consumer hardware) |
| KV cache sharding | Per KV-head per GPU | Single device for decode |
| Multi-node | Pipeline parallel via Ray | Not supported |

**[DEEP DIVE]** llama.cpp's `--tensor-split` is more accurately described as "layer distribution across GPUs" rather than true tensor parallelism. Different layers are mapped to different GPUs, and the activation is moved between GPUs as the forward pass traverses layers. This is closer to pipeline parallelism with micro-batch size 1. The implication is that consumer multi-GPU llama.cpp setups are PCIe-bound and gain throughput primarily from fitting a larger model rather than from parallelism speedup.

### When llama.cpp multi-GPU helps

- **Model capacity**: A GGUF Q4 Llama-3-70B is ~40 GB. Two 24 GB GPUs can hold it; one cannot.
- **Prefill speed**: Prefill is compute-bound and does benefit from GPU parallelism.
- **Decode speed**: Limited by inter-GPU data movement (PCIe). Marginal at best.

---

## 15.9 When to Add GPUs vs When to Replicate

The decision is not always "use TP". Two equally valid strategies exist:

### Strategy A: Tensor Parallel (scale-up)

One large model instance spread across N GPUs.

```
Pros:
  - Single model fits in HBM
  - Decode latency scales down (more bandwidth, parallel GEMM)
  - KV pool grows (more HBM available)

Cons:
  - AllReduce latency tax per step
  - All GPUs must be on same NVLink domain for reasonable performance
  - Single failure kills the instance
  - Scheduling complexity (N GPUs must be co-located)
```

### Strategy B: Replica Pool (scale-out)

N independent single-GPU instances behind a load balancer.

```
Pros:
  - Zero AllReduce overhead
  - Linear throughput scaling
  - Fault isolation (one GPU fails, N-1 instances still serve)
  - Works across PCIe or even across nodes

Cons:
  - Model must fit on a single GPU
  - Decode latency unchanged (still single-GPU-bound)
  - Total KV pool = N × single_GPU_kv (same total, but no cross-request sharing)
  - No single long context can use multiple GPUs' memory
```

**Decision heuristic:**

```
If model_weight_bytes > single_GPU_HBM × 0.8:
    → Must use TP (model doesn't fit)

Else if peak_concurrent_users × avg_prompt_tokens > single_GPU_kv_capacity:
    → Use TP to grow KV pool, OR use replica pool

Else if TTFT SLA < 200ms AND avg_prompt > 4K tokens:
    → Use TP (prefill parallelism cuts TTFT)

Else:
    → Replica pool (simpler, more fault-tolerant, same throughput)
```

### Worked comparison: 8B model, 8 GPUs, A100-40

```
Strategy A: TP=8 on 8 × A100-40
  Weights per GPU:    2 GB (16 GB / 8)
  KV pool per GPU:   ~33 GB (90% × 40 - 2 - 1.5) / per-token 16K B = 2.1M tokens
  AllReduce overhead: ~5% (NVLink)
  Decode latency:    ~1 ms/step (8× bandwidth)
  Max concurrency:   2.1M tokens / 8192 ctx = ~256 concurrent sessions

Strategy B: 8 replicas on 8 × A100-40
  Weights per GPU:   16 GB
  KV pool per GPU:   ~20 GB = 160K tokens per replica
  AllReduce overhead: 0
  Decode latency:    ~8 ms/step (single GPU)
  Max concurrency:   8 × (160K / 8192) = 8 × 19 = ~152 concurrent sessions
  Throughput:        8× (8 independent engines)
```

For this 8B case, **replica pool** wins on throughput (8 engines serving independently) while TP wins on per-session concurrency (more total KV tokens). TP also wins on decode latency for real-time latency-sensitive products.

---

## 15.10 quantization + TP Interaction

Quantised models complicate TP because quantization formats are not always TP-aware:

- **AWQ / GPTQ (vLLM):** supported with TP. Weight shards are dequantised on each GPU before the GEMM.
- **GGUF Q4_K_M (llama.cpp):** supported with `--tensor-split`. Dequantization is per-GPU.
- **FP8 (H100 native):** supported with TP in vLLM ≥ 0.4.

**[COMMON TRAP]** Quantising a 70B model to Q4 (≈40 GB) and trying to run on a single A100-80 is tempting — 40 GB fits! But the KV cache for 70B at full precision still uses 327 KB/token. With the Q4 model taking 40 GB and 90% of 80 GB usable = 72 GB, only 30 GB remains for KV. At 327 KB/token that is ~95,000 tokens — enough for about 11 sessions at 8K context. Adding TP=2 with Q4 gives each GPU 20 GB of weights and ~50 GB KV = ~306,000 tokens total, supporting ~37 sessions. The quantization and TP benefits compound.

---

## 15.11 Practical Configuration Walk-Through

### Case 1: Llama-3-8B, latency-sensitive chat, 4 × A100-40

```yaml
# vLLM config
model: meta-llama/Meta-Llama-3-8B-Instruct
tensor_parallel_size: 4          # 4-way TP: 4 GB weights/GPU, 32 GB KV/GPU
max_num_seqs: 256
max_num_batched_tokens: 32768    # large budget: prefill parallelized
max_model_len: 16384
block_size: 16
gpu_memory_utilization: 0.90
enable_chunked_prefill: true
enable_prefix_caching: true
```

Why TP=4? The 8B model fits on a single A100-40 (16 GB weights), but TP=4:

- Cuts decode latency 4× (bandwidth-bound: 4 GPUs read weights in parallel)
- Grows KV pool to ~128 GB total (4 × ~32 GB) → 1M+ tokens of context
- Gives headroom for 256 concurrent sessions at 4K context each

### Case 2: Llama-3-70B, RAG, 4 × A100-80 (NVLink)

```yaml
model: meta-llama/Meta-Llama-3-70B-Instruct
tensor_parallel_size: 4
max_num_seqs: 32
max_num_batched_tokens: 131072   # very large: RAG retrieval prompts are long
max_model_len: 65536
block_size: 16
gpu_memory_utilization: 0.92
enable_chunked_prefill: true
enable_prefix_caching: true
```

TP=4 is the minimum for 70B on A100-80 with usable KV pool (~35 GB/GPU after weights). 4× 35 GB = 140 GB KV total. At 327 KB/token (after TP sharding across 4 GPUs: 81.9 KB/token per GPU, but total pool is 4 × 35 GB = 140 GB shared): 140 × 10^9 / 81,920 ≈ 1.7M tokens. With max_model_len=65536: supports ~26 concurrent 64K sessions — perfect for RAG.

### Case 3: Llama-3-8B, offline batch, 8 × consumer 4090

```bash
# llama.cpp: equal split across 8 GPUs (each 24 GB)
llama-cli \
  --model llama-3-8b-q4_k_m.gguf \
  --n-gpu-layers 32 \
  --tensor-split 1,1,1,1,1,1,1,1 \
  --ctx-size 4096 \
  --parallel 8 \
  --ubatch-size 512 \
  --cache-prompt \
  --batch-size 4096
```

For batch inference on consumer hardware, llama.cpp with tensor-split gives you enough memory (8B Q4 = 4.5 GB; fits 5× on a single 24 GB 4090, but splitting across 8 GPUs leaves 19+ GB per GPU for KV). PCIe AllReduce penalty is mitigated by large batch sizes that amortize the fixed synchronization cost.

---

## 15.12 Failure Modes and Monitoring

### NCCL timeout

AllReduce requires all N GPUs to participate. If one GPU stalls (thermal throttle, driver issue, or memory error), the NCCL call hangs until the configurable timeout:

```
export NCCL_TIMEOUT=1800   # seconds; vLLM default is 1800s
```

A hung NCCL collective will block the entire serving engine. Monitor with:

```bash
nvidia-smi dmon -s u    # GPU utilization; a stalled GPU shows 0% util
```

### NCCL deadlock with async prefill

**[COMMON TRAP]** vLLM's async output processing can interact with NCCL in subtle ways when `--tensor-parallel-size > 1`. If async engine callbacks attempt to re-enter the TP communication group before the previous AllReduce completes, a deadlock occurs. The symptom is all GPUs stuck at 100% for minutes without output. Fix: ensure `--disable-async-output-proc` is not used in combination with aggressive async pipeline settings.

### Memory imbalance across TP ranks

If the TP split is unequal (custom `--tensor-split` in llama.cpp or asymmetric GPU memory), the GPU with the least free memory after weight loading determines the KV pool ceiling for that rank. vLLM uses the minimum across all TP ranks.

---

## 15.14 Expert Parallelism for MoE Models

Mixture-of-Experts (MoE) models replace the dense FFN sub-layer with a router and N specialist FFN "experts". At inference time only K of those N experts execute for each token — dramatically reducing per-token FLOPs — but all N expert weight matrices must remain in memory. This creates a unique parallelism challenge: no single GPU can hold all experts for a large model, and different tokens route to different experts, so compute cannot be trivially column/row-sharded the way dense TP works.

**Expert parallelism (EP)** solves this by distributing expert FFN blocks across GPUs: with EP=8 on a 256-expert model, each GPU holds 32 experts. Tokens are dispatched to the GPUs that hold their assigned experts via an **all-to-all** collective — each GPU sends tokens out to experts it does not hold, and receives tokens back that are destined for its experts.

### Token Routing Overhead

Top-K routing (K=2 is standard) means each token triggers 2 expert FFN computations on potentially 2 different GPUs. The communication pattern per forward pass step is:

```
Forward:
  1. Router (every GPU, in TP context): compute expert scores, select top-K experts
  2. All-to-all dispatch: each GPU sends its batch of tokens to the GPUs holding
     the chosen experts (one message per source-GPU/destination-GPU pair)
  3. Expert FFN execution: each GPU runs its local expert FFN on the received tokens
  4. All-to-all combine: results are sent back to the originating GPU
  5. Weighted combine: expert outputs are multiplied by router weights and summed
```

The all-to-all has the same bandwidth requirement as an AllReduce of equal message size but different latency profile — it requires EP×EP point-to-point transfers rather than a ring traversal. On NVLink this is fast; over InfiniBand it becomes the bottleneck.

### Load Imbalance Problem

A critical failure mode of expert parallelism is **load imbalance**: if the router consistently sends most tokens to a handful of popular experts, the GPUs holding those experts become the bottleneck while others sit idle.

Mitigations used in practice:

- **Auxiliary load-balancing loss**: added during training to penalise unequal expert utilization; encourages the router to spread tokens more evenly.
- **Expert capacity buffers**: each expert has a maximum token buffer (`capacity_factor`); tokens that overflow are either dropped or handled by a shared expert.
- **Expert dropping**: vLLM and DeepSpeed-MoE silently drop overflow tokens (with a log warning). This trades accuracy for latency predictability.
- **Shared expert**: DeepSeek-V2/V3 adds one always-selected "shared expert" per layer that all tokens use, providing a dense fallback path.

### Expert Parallelism in DeepSeek-V3

DeepSeek-V3 uses **256 routed experts + 1 shared expert** per MoE layer. With top-2 routing (each token selects 2 of the 256 routed experts), the routing pattern per token is:

```
DeepSeek-V3 EP configuration (from their technical report):
  Total experts:       256 routed + 1 shared
  EP degree:           EP=8 (32 routed experts per GPU)
  Routing:             top-2 of 256 routed experts
  Shared expert:       1 (processed on every GPU in parallel)
  All-to-all:          required for 2 of the 3 expert computations per token
```

The shared expert runs locally on every GPU — no all-to-all needed for it. Only the 2 routed expert lookups require cross-GPU all-to-all, making the shared expert a free computation that stabilises training and inference.

### Worked Example: Token Distribution Variance

Setup:

- 256 experts, EP=8 (32 experts/GPU), top-2 routing, batch=64 tokens

Expected tokens per expert (uniform routing):
```
  2 activations per token × 64 tokens = 128 activations total
  128 / 256 experts = 0.5 tokens per expert on average
  Per GPU (32 experts): 0.5 × 32 = 16 tokens expected
```

With uniform routing, each GPU processes 16 tokens — ideal load balance.

In practice, popular experts receive 3–5× the average load. A GPU with 5× load serves 80 tokens; a GPU with 0.2× load serves ~3 tokens. The slowest GPU determines the step latency, so variance directly hurts throughput.

Communication cost estimate (NVLink, H100 SXM, 600 GB/s bidirectional):
```
  Token embedding: d_model = 7168 (DeepSeek-V3), BF16 = 2 bytes
  Bytes per token: 7168 × 2 = 14 336 bytes ≈ 14 KB
  Tokens dispatched per GPU: 64 tokens × 2/8 = 16 tokens (2 routed, spread across 8)
  Dispatch bytes per GPU: 16 × 14 KB = 224 KB
  All-to-all time: 224 KB / (600 GB/s / 8 peers) ≈ 224 KB / 75 GB/s ≈ 3 µs
```

At this scale, the all-to-all communication cost is well under 10 µs on NVLink — negligible compared to the FFN GEMM itself. Over PCIe or InfiniBand the cost rises to hundreds of microseconds.

### vLLM Expert Parallelism Configuration

```bash
# vLLM expert parallelism for MoE models
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --expert-parallel-size 8 \
  --trust-remote-code \
  --max-num-seqs 32

# EP and TP can be combined: EP handles expert sharding,
# TP handles attention weight sharding within each expert group.
# Effective parallelism = TP × EP across the full GPU cluster.
```

The `--expert-parallel-size` flag (added in vLLM ≥ 0.5) distributes expert FFN blocks across the EP group. Each GPU in the EP group holds `total_experts / EP` expert weight matrices.

---

## 15.15 Pipeline Bubble Analysis and Microbatching

Pipeline parallelism (PP) assigns consecutive transformer layers to consecutive pipeline stages. A model with L layers and P stages assigns L/P layers per stage. Unlike TP, PP requires no high-bandwidth interconnect — it only passes activation tensors between adjacent stages, making it suitable for multi-node deployments over InfiniBand.

### The Pipeline Bubble

The fundamental inefficiency of pipeline parallelism is the **pipeline bubble**: during startup and teardown, earlier stages are idle while later stages work, and vice versa.

For a synchronous pipeline with P stages and M microbatches:

```
Bubble fraction = (P - 1) / (P - 1 + M)
```

This formula counts the idle stage-steps as a fraction of total stage-steps. As M → ∞, the bubble fraction → 0; as M → 1, the bubble fraction → (P-1)/P.

### Worked Examples

```
Setup: P = 4 pipeline stages

M = 1  microbatch:  bubble = (4-1) / (4-1 + 1)  = 3/4  = 75.0%
M = 2  microbatches: bubble = (4-1) / (4-1 + 2)  = 3/5  = 60.0%
M = 4  microbatches: bubble = (4-1) / (4-1 + 4)  = 3/7  = 42.9%
M = 8  microbatches: bubble = (4-1) / (4-1 + 8)  = 3/11 = 27.3%
M = 16 microbatches: bubble = (4-1) / (4-1 + 16) = 3/19 = 15.8%
M = 32 microbatches: bubble = (4-1) / (4-1 + 32) = 3/35 =  8.6%
```

This is why PP requires large batch sizes to be efficient: a 4-stage pipeline at M=1 wastes 75% of its compute time in bubbles. At M=16 the waste falls to 16% — similar to the 5–10% overhead of NVLink-based TP.

### vLLM and Pipeline Parallelism with Continuous Batching

vLLM's continuous batching architecture interacts with PP in a specific way: **each scheduler batch is effectively a microbatch**. The scheduler emits one batch per iteration; that batch flows through all pipeline stages sequentially.

```
PP stage flow for a single vLLM scheduler batch:
  Iteration i:   Stage 1 processes batch[i]
                 Stage 2 processes batch[i-1]   (from previous iteration)
                 Stage 3 processes batch[i-2]
                 Stage 4 processes batch[i-3]

  At steady state, all stages are busy (different batches).
  The bubble only appears during the first P-1 warm-up iterations
  and the last P-1 drain iterations.
```

Because vLLM's batches are continuous (there is always a next batch from the scheduler), the pipeline reaches steady state quickly and the per-batch bubble is small in practice. The more relevant concern is **inter-stage latency**: the activation tensor must transfer between stages, and this transfer is on the critical path.

```
Inter-stage activation size (Llama-3-8B, batch=32 sequences, 1 token/seq):
  Shape: [32, 1, 4096] in BF16
  Size: 32 × 1 × 4096 × 2 bytes = 262 KB

  NVLink (600 GB/s): 262 KB / 600 GB/s = 0.43 µs (negligible)
  InfiniBand HDR (200 Gb/s = 25 GB/s): 262 KB / 25 GB/s = 10.5 µs per stage boundary

  For P=4 stages: 3 boundaries × 10.5 µs = 31.5 µs per step
  Decode step time for 8B at batch=32: ~2–5 ms
  PP overhead: 31.5 µs / 2 ms = ~1.6% overhead (acceptable)
```

For large batches and small activation tensors, PP over InfiniBand is viable. For single-token decode at batch=1, the per-boundary latency becomes a significant fraction of the step time.

### The 1F1B Schedule (Training Context)

The **one-forward-one-backward (1F1B)** schedule from GPipe/PipeDream is the standard training pipeline schedule: alternating forward and backward microbatch passes keeps all stages busy during the steady state. This is **not applicable to inference** (there is no backward pass), but understanding 1F1B illuminates why the inference bubble formula is simpler: inference pipelines have no backward pass to interleave, so the bubble reduction must come entirely from M > 1.

For inference-only pipelines, the relevant schedule is the **all-forward-then-drain** pattern, and the bubble fraction formula above applies directly.

---



| Concept | Key fact |
|---------|----------|
| Why TP | Models > single-GPU HBM; decode bandwidth scales with GPU count |
| Sharding | Column-row split of MLP; head split of attention; 2 AllReduces per layer |
| AllReduce cost | NVLink: ~5% overhead; PCIe: up to 100% overhead |
| vLLM knob | `tensor_parallel_size=N`; must divide kv_heads |
| KV cache effect | TP=N halves KV bytes/token (KV heads sharded) |
| llama.cpp | `--tensor-split` is layer offload, not true Megatron TP |
| Scale-up vs out | TP when model doesn't fit or TTFT matters; replicas when throughput > latency |
| Break-even | PCIe TP breaks even only above ~40 tokens/batch for 8B models |

---

## Chapter Notes

**[FOUNDATIONAL]** The two AllReduces per transformer layer are not optional — they are baked into the Megatron-LM parallelism design that all modern TP implementations follow. Understanding them is prerequisite to any TP performance analysis.

**[DEEP DIVE]** The ring-allreduce algorithm is described in the original Baidu paper (Sergeev & Del Balso, 2018, "Horovod: fast and easy distributed deep learning in TensorFlow"). The key insight is that ring-allreduce is bandwidth-optimal: it achieves the theoretical minimum communication time for the given network topology.

**[COMMON TRAP]** Developers often assume `tensor_parallel_size=2` always helps. It does not when: (a) the model fits on one GPU, (b) the interconnect is PCIe, or (c) batch sizes are small. Measure first.

---

*Companion code: `code/chapter_15/multigpu_demo.py` and `code/chapter_15/multigpu_demo.cpp`*


---

## Chapter Summary

- **Tensor parallelism (TP)**: splits each weight matrix column-wise (up-projection) and row-wise (down-projection) across TP ranks; requires one AllReduce per matmul pair.
- **Pipeline parallelism (PP)**: assigns consecutive transformer layers to consecutive devices; requires one activation transfer per micro-batch per pipeline stage boundary.

> **LinkedIn Scenario Update:** LinkedIn's $1.2M/month cluster running at 28% utilization is likely serving a 70B-class model that cannot fit on a single GPU. Without tensor parallelism, the only option is model sharding via CPU offload — which is 10–20× slower and entirely unsuitable for 50K concurrent users. With TP=4 across NVLink-connected A100-80s, the 70B model fits in 35 GB per GPU (at BF16), leaving ~45 GB per GPU for KV cache. A single 4-GPU node can then serve ~30 concurrent sessions at 8K context — enough to handle the LinkedIn workload's latency SLA with far fewer nodes than a naive 2-GPU-per-model configuration, directly reducing the per-month cost.

- **TP vs PP trade-off**: TP requires high-bandwidth interconnect (NVLink) and low latency; PP can work over slower links but introduces pipeline bubble overhead.
- **Pipeline bubble formula**: bubble_fraction = (P-1) / (P-1 + M) for P stages and M microbatches; at P=4, M=1 the bubble is 75%; at M=16 it falls to 16%.
- **vLLM TP implementation**: `--tensor-parallel-size N` shards all attention and FFN weights; model parallelism uses NCCL AllReduce on the NVLink ring.
- **Worker process model**: vLLM spawns N worker processes (one per GPU) using `ray` or `multiprocessing`; each holds its shard and communicates via NCCL.
- **NVLink vs PCIe**: NVLink (600 GB/s bidirectional on H100 SXM) is 10× faster than PCIe 4.0 (64 GB/s); TP efficiency drops sharply without NVLink.
- **Expert parallelism (EP)**: for MoE models, experts are distributed across GPUs via `--expert-parallel-size`; each token is routed to its top-K experts on potentially different devices via all-to-all communication.
- **Load imbalance in EP**: popular experts receive 3–5× average token load; auxiliary load-balancing loss during training and expert capacity buffers during inference mitigate this.
- **DeepSeek-V3 EP**: 256 routed experts + 1 shared expert, EP=8 (32 experts/GPU), top-2 routing; shared expert runs locally with no all-to-all overhead.

---

## Self-Check Questions

1. Tensor parallel size is 4. A weight matrix W is of shape (4096, 16384). Draw the column-parallel shard each rank holds. After the matmul on each rank, what AllReduce operation is needed? *(Section 15.2)*

2. An AllReduce on 4 A100 GPUs connected via NVLink takes 0.05 ms. The matmul it follows takes 0.8 ms. Compute the TP efficiency (useful compute / total time) for a 32-layer model. *(Section 15.3)*

3. Pipeline parallelism with 4 stages and a micro-batch size of 1 has a pipeline bubble of (P-1)/P = 75%. How do you reduce this? What is the bubble with 8 micro-batches? *(Section 15.4)*

4. You have 8 GPUs connected in a NVLink ring where each pair has 600 GB/s bidirectional. A ring AllReduce for a 256 MB gradient tensor has a theoretical floor of (2×(N-1)/N × bytes) / bandwidth. Compute this for N=8. *(Section 15.3)*

5. DeepSeek-V3 uses 256 experts with top-8 routing. With expert parallelism on 64 GPUs, describe the all-to-all communication pattern: how many messages are sent per forward pass step per token? *(Section 15.5)*

6. A pipeline has P=6 stages. You want to keep the bubble fraction below 20%. What is the minimum number of microbatches M required? Show your work using the bubble formula. *(Section 15.15)*

7. In an MoE model with 128 experts, EP=4, top-2 routing, and batch=128 tokens, compute the expected number of tokens each GPU's experts process. If the 10 most popular experts each receive 12 tokens while all others receive the average, what is the maximum per-GPU token load and how does it affect step latency? *(Section 15.14)*


---

## Worked Solutions

### Question 1
**Setup:** TP=4, weight matrix W of shape (4096, 16384).

**Column-parallel sharding:**
The weight is split along the output dimension (columns). Each rank holds:
```
W_rank = shape (4096, 4096)   # 16384 / 4 columns each
```

**After the matmul:**
Each rank computes `x @ W_rank`, producing a partial output of shape `(batch, 4096)`.

These partial outputs must be concatenated (not summed), because column-parallel splits the output features — each rank produces a different subset of output neurons. The operation needed is **AllGather**, not AllReduce. The four partial outputs are gathered so every rank has the full `(batch, 16384)` result.

*Exception:* For the row-parallel layer that follows (which splits the input dimension), each rank receives its own slice of the input and the partial products are summed — that layer requires **AllReduce**.

---

### Question 2
**Setup:** AllReduce = 0.05 ms, matmul = 0.8 ms, 32 layers.

**Per-layer compute time:**
```
t_layer = matmul + AllReduce = 0.8 + 0.05 = 0.85 ms
```

**TP efficiency per layer:**
```
efficiency = t_useful / t_total = 0.8 / 0.85 ≈ 94.1%
```

**For 32 layers:**
All layers are sequential. The efficiency is the same at each layer:
```
total useful   = 32 × 0.8  = 25.6 ms
total elapsed  = 32 × 0.85 = 27.2 ms
overall TP efficiency = 25.6 / 27.2 ≈ 94.1%
```

This 5.9% overhead is acceptable on NVLink. On PCIe, AllReduce cost rises to ~0.5 ms per layer, dropping TP efficiency to 0.8/1.3 ≈ 61.5% — a strong reason to avoid high TP on PCIe nodes.

---

### Question 3
**Pipeline bubble with P=4, micro-batch=1:**

**Bubble fraction formula:** `(P − 1) / P = 3/4 = 75%`

This means 75% of GPU time is wasted on pipeline fill and drain (GPUs idle while the pipeline fills up at startup and drains at the end).

**Reducing the bubble:**
Use more micro-batches (M). Bubble fraction becomes:
```
bubble = (P − 1) / (M + P − 1)
```

**With M=8, P=4:**
```
bubble = (4 − 1) / (8 + 4 − 1) = 3 / 11 ≈ 27.3%
```

This is a 2.75× reduction. In practice, M=8–16 micro-batches are used to push bubble fraction below 20%, which requires M ≥ 4(P−1) = 12 for sub-20% at P=4.

---

### Question 4
**Setup:** N=8 GPUs, NVLink 600 GB/s bidirectional, tensor = 256 MB.

**Ring AllReduce formula:**
```
time = 2 × (N − 1)/N × bytes / bandwidth
```

Substituting:
```
time = 2 × (7/8) × 256 MB / 600 GB/s
     = 2 × 0.875 × (256 × 10⁻³ GB) / 600 GB/s
     = 2 × 0.875 × 4.267 × 10⁻⁴ s
     = 7.467 × 10⁻⁴ s
     ≈ 0.747 ms
```

**In context:** A typical forward pass matmul for a 70B model layer at batch=32 takes ~5–10 ms. A 0.75 ms AllReduce is ~7–15% overhead — significant but tolerable on NVLink. On PCIe at 64 GB/s, the same AllReduce takes ~7 ms, nearly equalling the compute time and making TP unattractive.

---

### Question 5
**Setup:** 256 experts, top-8 routing, EP=64 GPUs.

**Expert assignment:** 256 experts / 64 GPUs = 4 experts per GPU.

**All-to-all pattern:**
Every GPU holds tokens that need to be routed to 8 experts. Since experts are distributed across GPUs:

Per token: top-8 experts selected → each expert may be on a different GPU.
Per forward pass step: each GPU sends token activations to up to 8 other GPUs and receives tokens from up to 8 other GPUs.

**Messages sent per token per step:**
A token is processed on 8 expert GPUs → 8 send operations (one per expert). Each send carries the token's hidden-state activation vector (d_model floats).

**Total messages in the system per step:**
With batch B tokens distributed across 64 GPUs, each GPU sends ≤ B/64 × 8 messages = up to 128 messages per GPU per step (2 all-to-all collectives: dispatch + combine).

This all-to-all is the primary bottleneck for MoE scaling. NVLink or InfiniBand with RDMA is required; PCIe cannot sustain the necessary bandwidth at scale.

---

### Question 6
**Bubble formula:** `bubble = (P − 1) / (M + P − 1) < 0.20`

Solve for M with P=6:
```
(6 − 1) / (M + 6 − 1) < 0.20
5 / (M + 5) < 0.20
M + 5 > 25
M > 20
```
**Minimum M = 21 micro-batches.**

Verification: bubble = 5 / (21 + 5) = 5/26 ≈ 19.2% < 20% ✓

---

### Question 7
**Setup:** 128 experts, EP=4, top-2 routing, batch=128 tokens.

**Average tokens per expert:** Each token routes to 2 experts. Total expert "visits" = 128 × 2 = 256. With 128 experts:
```
average tokens per expert = 256 / 128 = 2 tokens
```

**Experts per GPU:** 128 / 4 = 32 experts/GPU.

**Average tokens per GPU:** 32 × 2 = 64 tokens.

**Load imbalance:** The top 10 experts each receive 12 tokens. If these 10 experts are on 1–2 GPUs (worst case), one GPU processes:
```
max load = 10 × 12 + 22 × 2 = 120 + 44 = 164 tokens
```
vs. average of 64 tokens → **2.56× load imbalance**.

**Effect on step latency:** The step cannot complete until the most-loaded GPU finishes. At 2.56× overload, step latency is effectively 2.56× the average-case latency — even though 3 of 4 GPUs are sitting idle waiting. This is the "expert collapse" problem. Mitigation strategies: auxiliary load-balancing loss during training, token dropping with capacity factor < 1.0, or expert parallelism rebalancing.

