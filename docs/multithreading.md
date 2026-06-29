# Multithreaded Matrix Multiplication

## Choosing `std::thread`

Several STL concurrency primitives were evaluated:

| Primitive | Pros | Cons | Verdict |
|---|---|---|---|
| `std::thread` | Fine-grained control, no external deps, works with C++11+ | Manual join management | **Chosen** |
| `std::async` / `std::future` | Cleaner API, exception propagation | Implicit future destruction can block; less predictable scheduling | Rejected — hidden sync surprises |
| `std::execution::par` (C++17) | One-liner with parallel algorithms | Requires TBB or PSTL backend; not universally available | Rejected — portability |
| OpenMP `#pragma omp parallel for` | Very concise | Non-standard, requires compiler flag, brittle with nested parallelism | Rejected — stick to pure STL |

`std::thread` gives us explicit control over work partitioning without hidden synchronization costs.

---

## Work Partitioning

The output matrix $C_{M \times N}$ is split **row-wise** across threads:

```
Thread 0  →  rows [0,          rowsPerThread)
Thread 1  →  rows [rowsPerThread, 2×rowsPerThread)
  ...
Thread T-1 →  rows [t×rowsPerThread, M)
```

Row-wise splitting is ideal because:

- Each output row $C[i][:]$ depends on row $A[i][:]$ and the entire $B$ matrix
- Threads only read from shared $A$ and $B$ (no writes → no false sharing)
- Threads write to disjoint output rows (no mutex needed)
- No synchronization required except `join()` at the end

### Thread count logic

```cpp
unsigned int numThreads = std::thread::hardware_concurrency();  // physical cores
if (numThreads == 0) numThreads = 4;   // fallback if detection fails
if (numThreads > M)  numThreads = M;   // don't spawn more threads than rows
if (numThreads < 1)  numThreads = 1;   // safety floor
```

---

## Loop Ordering: $i \to k \to j$

### Before (original): $i \to j \to k$

```cpp
for (i = 0; i < M; ++i)
    for (j = 0; j < N; ++j) {
        float acc = 0;
        for (k = 0; k < K; ++k)
            acc += A[i][k] * B[k][j];   // B access: strideB0 jump each iteration
        C[i][j] = acc;
    }
```

**Problem**: The innermost loop accesses $B[k][j]$ — for each $k$ increment, the pointer jumps by `strideB0` (which is $N$ for row-major). This evicts cache lines constantly.

### After (optimized): $i \to k \to j$

```cpp
for (i = 0; i < M; ++i) {
    zero(C[i][:]);
    for (k = 0; k < K; ++k) {
        float aik = A[i][k];                  // loaded once
        const float* B_row = B[k][:];          // single row pointer
        for (j = 0; j < N; ++j)
            C[i][j] += aik * B_row[j];         // sequential streaming
    }
}
```

**Why it's better**:

1. **$A[i][k]$ loaded once** per $(i,k)$ pair — hoisted out of the $j$ loop
2. **$B[k][:]$ streamed sequentially** — prefetcher can keep up, full cache line utilization
3. **$C[i][:]$ stays in L1 cache** — accumulated in-place across $k$, single cache line per row element

Memory access pattern (assuming row-major, contiguous strides):

| Original ($i \to j \to k$) | Optimized ($i \to k \to j$) |
|---|---|
| $B$: $[k \cdot N + j]$ — stride-$N$ jumps | $B$: sequential `B_row[j]` |
| $A$: sequential | $A$: single load per $(i,k)$ |
| $C$: written once per $(i,j)$ | $C$: read-modify-write, stays hot |

---

## Pointer Hoisting

Stride values are captured in local variables to avoid repeated struct field access:

```cpp
uint64_t strideA0 = a.stride[0], strideA1 = a.stride[1];
uint64_t strideB0 = b.stride[0], strideB1 = b.stride[1];
uint64_t strideO0 = out.stride[0], strideO1 = out.stride[1];
```

Row base pointers are hoisted out of the innermost loop:

```cpp
float*       poRow  = po + i * strideO0;        // computed once per row
const float* pbRow  = pb + k * strideB0;        // computed once per (i,k)
```

This eliminates $\mathcal{O}(M \times K \times N)$ pointer arithmetic inside the hot loop.

---

## Synchronization Model: Fork-Join

```
main thread
  │
  ├─ spawn Thread 0 ──┐
  ├─ spawn Thread 1 ──┤  all run in parallel
  ├─ spawn Thread 2 ──┤  (no shared mutable state)
  ├─ ...              ──┘
  │
  ├─ join() Thread 0  ← blocks until done
  ├─ join() Thread 1
  ├─ join() Thread 2
  └─ return            (all output rows fully computed)
```

No mutexes, no atomics, no condition variables. The only synchronization is the implicit barrier at `join()`.

### Why no false sharing?

Each thread writes to a **contiguous block of rows** in the output matrix. Thread 0 writes `C[0..r-1][:]`, Thread 1 writes `C[r..2r-1][:]`, etc. Since each thread's output region is a multiple of the row size (typically much larger than a cache line), the write regions don't share cache lines.

---

## Full Worker Lambda

```cpp
auto worker = [&](uint64_t iStart, uint64_t iEnd) {
    for (uint64_t i = iStart; i < iEnd; ++i) {
        float* poRow = po + i * strideO0;

        // Zero this output row before accumulation
        for (uint64_t j = 0; j < N; ++j)
            poRow[j * strideO1] = 0.0f;

        // Outer-product accumulation
        for (uint64_t k = 0; k < K; ++k) {
            float aik = pa[i * strideA0 + k * strideA1];
            const float* pbRow = pb + k * strideB0;
            for (uint64_t j = 0; j < N; ++j) {
                poRow[j * strideO1] += aik * pbRow[j * strideB1];
            }
        }
    }
};
```

---

## Launch & Join

```cpp
std::vector<std::thread> threads;
threads.reserve(numThreads);
uint64_t rowsPerThread = (M + numThreads - 1) / numThreads;  // ceil division

for (unsigned int t = 0; t < numThreads; ++t) {
    uint64_t start = t * rowsPerThread;
    uint64_t end   = std::min(start + rowsPerThread, M);
    if (start >= end) break;
    threads.emplace_back(worker, start, end);
}

for (auto& thr : threads)
    thr.join();
```

`emplace_back` constructs the `std::thread` in-place, forwarding `start` and `end` to the worker lambda. The `join()` loop ensures all threads complete before `mulTensors` returns.

---

## Performance Characteristics

| Factor | Impact |
|---|---|
| **Row-wise split** | No false sharing, zero mutex overhead |
| **$i \to k \to j$ ordering** | Sequential B access, full cache line utilization |
| **Pointer hoisting** | Eliminates repeated stride multiplication |
| **Thread count** | Capped at `min(hardware_concurrency, M)` — no oversubscription |
| **Scalability** | Linear with rows $M$; for small $M$, falls back to single-threaded naturally |

### When multithreading doesn't help

For very small matrices ($M < 4$, small $N$), thread creation overhead may outweigh the parallelism benefit. The `numThreads > M` guard ensures we don't over-partition. For future optimization, a threshold check (e.g., only multithread if $M \times K \times N > 10^5$) could be added.
