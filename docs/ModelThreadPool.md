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

- **Size.** The pool has as many threads as the largest parallel region so
  far asked for (`num_threads` of the spec), and at most one per CPU the
  process may run on (`BUDDY_THREAD_POOL_THREADS` lowers the limit). The first
  region starts the pool; a region that asks for more threads adds them. Each
  thread is pinned to its CPU, and so is the first caller.
- **Waiting.** The threads wait for work by spinning. A region therefore starts
  and ends a few microseconds sooner than with libomp, which matters when a
  decode step runs a few hundred short regions. After about 16 M spins without
  work, a thread sleeps 50 µs between checks.
- **Scheduling.** Only static schedules are supported, which is what
  `omp.wsloop` uses without a schedule clause.
- **Nesting.** Nested parallel regions run serially, as they do with libomp by
  default.
- **Concurrency.** Parallel regions that different application threads start
  wait for each other. The thread count an application thread asks for
  (`__kmpc_push_num_threads`) is kept per thread and applies to its own next
  region.
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

## Profiling

With `BUDDY_THREAD_POOL_PROFILE=1`, the pool times every parallel region and
prints the totals to stderr when the model library is unloaded:

- the time from the first region to the last, the time in regions, the gaps
  between regions (it is their sum), and the barrier waits of all threads;
- per region function (`<library>+<offset>`; `nm -n` on the library names
  it): the calls, the wall time, the gaps before and after the region, the
  barrier waits of all threads and of thread 0, and the barriers.

A gap is all the wall time from the end of one region to the start of the
next: the caller working alone (serial code between parallel loops) while
the other threads wait, but also any time the process spends outside the
model between two calls (sampling, I/O, a server waiting for requests). It
measures serial work only in a run that calls the model back to back, such
as `buddy-cli` generating text. The columns before and after attribute the
same gaps to the region that follows and to the one that precedes them: do
not add them up.

Barrier waits measure load imbalance: the threads that finish their share
first wait for the last. Per token figures come from the difference between
two runs that generate different numbers of tokens.

The table has 1024 slots for region functions; the regions of any further
function are counted together as `other+0x0`, with a warning.

Timing adds two clock reads per region and per barrier, and an atomic
addition to a counter that all threads share per barrier. On the K3 with 8
threads, this is under 1% of a Qwen3-VL-30B-A3B decode step (337 regions per
token); with more threads or shorter regions it can be more. Without the
variable, the pool costs one branch per region and per barrier.

## Tests

`tests/Python/test_buddy_thread_pool.py` lowers parallel loops as
`compile_pipeline.py` does and links them with the pool and without libomp;
this only links if the pool provides every entry point the loops call. It then
runs them and checks that:

- every iteration runs once, on the expected number of threads;
- `BUDDY_THREAD_POOL_THREADS` lowers that number;
- a nested parallel loop computes the right result;
- `BUDDY_THREAD_POOL_PROFILE=1` gives the same results and a profile with the
  call count of each region function, whose time from the first region to
  the last is the time in regions plus the gaps, also with two application
  threads; with fewer slots than region functions, the extra ones are
  reported as `other`, with a warning;
- two application threads that ask for 2 and 3 threads get their own count,
  with the interleaving push A, push B, fork A, fork B forced, and over 2000
  concurrent calls each.
