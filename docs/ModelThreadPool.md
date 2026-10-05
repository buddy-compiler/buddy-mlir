# Model Thread Pool

The parallel loops of a `tools/buddy-codegen` model are lowered to OpenMP
(`-convert-scf-to-openmp`) and call libomp. With the spec field

```json
{
  "thread_pool": true
}
```

the model library runs them on a thread pool of its own instead,
`runtime/threadpool/BuddyThreadPool.c`, and does not link libomp. It is off by
default.

## What the pool does

The generated code calls six libomp entry points: `__kmpc_fork_call`,
`__kmpc_push_num_threads`, `__kmpc_global_thread_num`, `__kmpc_barrier`,
`__kmpc_for_static_init_8u` and `__kmpc_for_static_fini`. The pool implements
these and nothing else. Its symbols are hidden, so the model library binds to
them when it is linked.

- **Startup.** The first parallel region starts the pool. It creates as many
  threads as the region asks for (`num_threads` of the spec), with at most one
  per CPU the process may run on (`BUDDY_THREAD_POOL_THREADS` lowers the
  limit). Each thread is pinned to its CPU, the calling thread included.
- **Waiting.** The threads wait for work by spinning. A region therefore starts
  and ends a few microseconds sooner than with libomp, which matters when a
  decode step runs a few hundred short regions. After about 16 M spins without
  work, a thread sleeps 50 µs between checks.
- **Scheduling.** Only static schedules are supported, which is what
  `omp.wsloop` uses without a schedule clause.
- **Nesting.** Nested parallel regions run serially, as they do with libomp by
  default.
- **Concurrency.** Parallel regions that different application threads start
  wait for each other.
- **Stack size.** `OMP_STACKSIZE` sets the stack of the pool threads. The
  default is 256 MiB of virtual memory.

The spinning threads keep their CPUs busy while the model runs. The pool suits
a machine that serves the model, not one shared with other work. With
`--is-rvv-crosscompile`, a model with `thread_pool` needs no
`--riscv-omp-shared`, and its `.rax` does not carry libomp. Layer partitioning
is not supported with this option.

## Results

DeepSeek-R1-Distill-Qwen-1.5B, `buddy-cli`, greedy decoding. The generated
text is the same with and without the pool.

| Build | Prompt | Prefill | Decode |
| --- | --- | --- | --- |
| SpacemiT K3, `w4g32` (8 A100 cores), libomp | 458 tokens | 9.87 s | 22.5 tok/s |
| the same with `thread_pool` | 458 tokens | 9.51 s | 23.8 tok/s |
| SpacemiT K3, `w4g32`, libomp | short | | 25.2 tok/s |
| the same with `thread_pool` | short | | 26.65 tok/s |
| x86, f32, `prefill_chunk` 64 (48 threads), libomp | 458 tokens | 6.5 s | 7.9 tok/s |
| the same with `thread_pool` | 458 tokens | 5.9 s | 11.3 tok/s |

The K3 rows are the mean of 4 alternating runs. The x86 rows are the mean of
4 alternating runs on a shared Xeon Platinum 8575C, where single runs vary by
30%.

## Tests

`tests/Python/test_buddy_thread_pool.py` lowers parallel loops as
`compile_pipeline.py` does and links them with the pool and without libomp;
this only links if the pool provides every entry point the loops call. It then
runs them and checks that:

- every iteration runs once, on the expected number of threads;
- `BUDDY_THREAD_POOL_THREADS` lowers that number;
- a nested parallel loop computes the right result.
