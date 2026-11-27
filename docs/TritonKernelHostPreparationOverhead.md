# Host preparation overhead on the RISC-V target

On 2026-11-27, four alternating strict/prepared/strict/prepared measurement groups completed on the actual RISC-V target with exit code 0. The prior [correctness replay](TritonKernelHostPreparationResults.md) had passed 56 RAX variants, 50 contract groups, and 1,000 Buddy comparisons. Each of the 12 timing processes again checked 1,000 complete-graph/raw-kernel comparisons and execution paths/status; all produced the tested-input digest `8791474098158242986`.

The complete dispatch increment fell 87.6% and prepared steady-state graph latency fell 57.2% relative to strict host checking. However, for this tiny 2×4 specialization the prepared Triton graph was still about 4.35% slower than the same-window Buddy graph, and first-session preparation became more expensive. No end-to-end speedup over Buddy is claimed.

## Platform and method

- Ubuntu only cross-compiled the target artifacts. All timing numbers below came from Linux riscv64 kernel 7.0.0-43-generic, glibc 2.43, RVV 1.0, VLEN 256; online/allowed CPUs 0–79. The timed process was pinned to CPU 0, with one operator thread.
- The measured Buddy source base was `0fa93b60fe961b0ae59c14492a42c2a88f4deadc` plus the documented uncommitted Issue #918 changes. Source/object digests identify the precise inputs. Both modes used the same binary, kernel object, link settings, `-O2` adapter/runtime, `rv64gc`, and LP64D. A startup environment choice selected strict or prepared checking.
- Strict checking retained the old per-invocation affinity/vector queries. Prepared checking validated all online CPUs once and kept a positive vector-permission cache in each actual execution thread. The call's tensor metadata was still checked each time.
- The input was contiguous FP32 `[[1,2,3,4],[0,0,0,0]]`, axis `-1`. Each group used three independent processes; each process ran ten alternating AB/BA rounds with 200,000 calls per metric per round. Four groups yielded 120 steady batches and 840 preparation samples, with six independent processes per mode and 60 batches per mode.
- Quantiles describe **batch mean** distributions, not single-call tail latencies; 60 batches are not 60 independent processes. Timing bookkeeping was included. OS page cache, clock frequency, and governor were not forcibly controlled; the governor/frequency interface was unavailable for inspection. Group environment snapshots were retained.

## The three Issue #918 metrics

Values are nanoseconds per call; parentheses show P25–P75. The table reports medians after pooling the two groups of each mode.

| Metric | Strict reference | Prepared checking | Interpretation |
| --- | ---: | ---: | --- |
| Kernel dispatch increment | 1,394.386 (1,388.685–1,401.457) | 173.392 (170.071–177.570) | Full graph minus raw Triton, paired **within each round** using the same object/library; down 87.6%. |
| Data-layout conversion | N/A | N/A | Existing contiguous row-major buffers were borrowed directly; no conversion occurred. N/A is not a measured 0 ns. |
| End-to-end Triton operator graph | 2,135.017 (2,129.206–2,146.616) | 913.263 (910.344–922.162) | Compiled graph entry to synchronous return, including adaptation, checks, dispatch, and the actual kernel; down 57.2%. |
| Same-window Buddy graph | 874.560 (872.319–877.197) | 875.176 (873.819–876.896) | Existing Buddy-generated computation plus its actual return adaptation. Prepared Triton/Buddy median ratio: 1.04352. |

The dispatch metric first subtracts each round's raw result from that round's graph result, then summarizes the differences. Subtracting two pooled medians would be a different statistic. It includes graph entry and validation, not just a function-pointer call. Prepared graph latency is about 2.34 times faster than the strict Triton graph, **not** 2.34 times faster than Buddy. The median per-round prepared Triton/Buddy ratio was 1.04442 (P25–P75 1.03998–1.05308; P95 1.12404). The graph median in every prepared process was higher than that process's Buddy reference.

| Supporting metric, median ns | Strict | Prepared |
| --- | ---: | ---: |
| Host compatibility callback | 1,166.551 | 49.592 |
| Raw Triton kernel path | 741.831 | 742.079 |
| Full checked adapter including kernel | 1,983.359 | 808.578 |
| Runtime dispatch call including kernel | 2,037.823 | 849.239 |

The compatibility callback fell about 95.7%, consistent with instrumented evidence that the monitored affinity/vector queries no longer repeat in steady state. The prepared callback still reads thread-local state and checks caches. Raw-kernel medians are close, supporting an integration-overhead explanation; the remaining 173 ns cannot be assigned to one function. It combines repeated call validation, memref/call adaptation, Runtime matching, adapter argument assembly, and timing checks.

| Group | Mode | Triton graph median ns | Buddy median ns | Paired dispatch increment ns |
| --- | --- | ---: | ---: | ---: |
| 0 | Strict | 2,129.923 | 875.755 | 1,390.164 |
| 1 | Prepared | 921.440 | 875.094 | 176.819 |
| 2 | Strict | 2,143.167 | 873.684 | 1,397.091 |
| 3 | Prepared | 912.115 | 875.176 | 172.366 |

One prepared process had a raw median of about 801.738 ns and a graph median of about 979.680 ns. It is retained in the results. Its paired dispatch increment was about 177.909 ns, near the other processes. The evidence does not isolate this raw variation to frequency, allocator behavior, or code layout.

## Cold and one-shot costs

Values are milliseconds unless noted; first-session and first-call entries have six independent processes per mode.

| Metric | Strict median | Prepared median |
| --- | ---: | ---: |
| Session preparation including load/query/capability checks | 6.040 | 11.940 |
| First external call after Session preparation | 0.0356 | 0.0420 |
| Preparation plus first call, summed within each process | 6.087 | 11.981 |
| Native RAX CLI whole-process wall time, 60 samples per mode | 18.651 | 25.607 |

The prepared mode creates a temporary worker thread and traverses 80 CPUs, increasing median Session preparation by about 5.9 ms. A one-shot operator or CLI run is therefore slower; steady-state gains apply only when the Session is reused. Seven file/payload and extraction-cache scenarios each had 60 samples per mode. `empty`/`existing` describe payload extraction cache state, not a cleared OS or hardware cache. No automatic profitability threshold or performance-based kernel selection was added.

## Copies and limits

- The integration borrows the original tensor pointers and adds no layout conversion or whole-tensor transfer buffer. This does **not** imply zero copies: the same Triton object internally stages 32 bytes across two rows and allocates temporary storage, while Buddy fallback makes a 32-byte return adaptation copy. These costs are already included in their respective end-to-end measurements and are not added again.
- Removing repeated host queries does not make the 2×4 Triton graph faster than Buddy. Repeated metadata checks and boundary adaptation can be assessed separately. Triton-internal storage belongs to Triton-RISCV compiler/kernel work; Buddy should not duplicate its lowering or replace the proof with hand-written Softmax.
- The result applies only to this shape/layout/target, not a whole model, other shapes, or streaming DRAM workloads.
- Four supplementary `perf` attempts were denied (exit code 255). No `sudo` or system setting change was used. The timing and correctness conclusions do not rely on `perf`, and no sample-attribution percentages are claimed.

## Data and reproduction

On the target, from the extracted measurement bundle:

```bash
python3 compare_host_modes.py --artifact-dir . --output ./host-mode-comparison --cpu <allowed-cpu>
```

The [example benchmark](../examples/TritonSoftmax/benchmark.py) and [A/B comparison driver](../examples/TritonSoftmax/compare_host_modes.py) reproduce the measurement protocol and write structured raw results plus per-group summaries. The 120 steady batches and 840 preparation samples from this run were retained in the local ignored build directory rather than added to the source PR. Complete transport logs, private machine paths, and local shell environment also remain there.
