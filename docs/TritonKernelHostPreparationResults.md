# Host preparation: build and correctness results

On 2026-11-27, all target C/C++/assembly artifacts were cross-compiled on Ubuntu and exercised on the RISC-V target. The separate [overhead report](TritonKernelHostPreparationOverhead.md) contains the subsequently measured timings. The kernel object and Buddy graph objects are the same as in the earlier validated build; the change relocates host checking rather than changing the Softmax arithmetic.

## Actual results

| Check | Result |
| --- | --- |
| Formatting/static checks on changed files | `pre-commit` passed for the checked set. |
| Cross artifacts | 39 riscv64 LP64D ELF files; the production kernel library exports only `buddy_external_kernel_query_v1`. |
| Fresh extracted bundle | 113 bundle hashes verified; final verifier exit code 0. |
| Standalone C++ invocation | Both rows of the fixed input matched the expected Softmax output. |
| Existing Buddy implementation | 1,000 finite inputs compared; maximum measured absolute difference 0, `atol=1e-6`, `rtol=1e-5`. |
| Auxiliary double oracle | Maximum absolute error `7.2829e-8`. |
| RAX | 56 variants passed, covering actual external execution, actual Buddy fallback, and expected errors. |
| Contracts | 50 groups passed: the original 47 plus three host-preparation groups. |
| Invalid arguments | 42 graph/runtime cases and 43 direct-adapter cases; input and output sentinels/canaries checked. |
| CPU domain and migration | Results checked on 80 verified compatible CPUs; multi-CPU affinity was accepted. |
| Permission cache | Thread isolation, failure retry, and eight concurrent first-use threads passed. |
| Preparation failure | Injected affinity failure caused 100 valid Buddy fallbacks; the external kernel did not run. |
| Strict/prepared comparison | Both modes produced output digest `8791474098158242986` over the tested inputs, using the same kernel and graph objects. |

The target ran Linux riscv64 kernel 7.0.0-43-generic and glibc 2.43, with online CPUs 0–79 and VLEN 256. The three kernel/graph object SHA-256 digests matched the previous validated artifacts. The reported bit digest applies to the tested inputs only; it does not claim universal bit identity.

The board verifier writes `board-summary.json`, `rax-results.json`, and `contract-results.json` in the extracted test bundle. The observed host-preparation checks were:

```text
PASS preparation queries_cached=1 worker_isolated=1 retry=1 caller_affinity_preserved=1 migration_cpus=80 steady_syscalls=0 fork_exec=1 on_to_off_eperm=1
PASS rejected domain fallback=100 output=valid
PASS concurrent first-use workers=8 independent_permission=1
```

`steady_syscalls=0` means zero additional affinity read/write and vector-policy queries in the instrumented steady window. It does not assert that the whole process has no system calls or allocations and is not a latency measurement.

## Failure investigations retained as limits

- The copied sysroot lacked a pthread development linker file. The test build used a task-local linker shim derived from the target's read-only runtime library; the original sysroot and target system were not modified.
- An early export record omitted `query_entry`. The corrected run used a previously validated full record and kept the same kernel object. Only the adapter deployment policy `requires_single_cpu` was migrated to `false`, with the original record and digest retained.
- A dynamic C++ test image with vector `NEXT=OFF` trapped with SIGILL during dynamic startup, before the adapter opened. This case is **not** recorded as a passing application path. A freestanding static probe without a dynamic loader, libc, or RVV instructions verified the exec policy (`PASS exec-off static image policy=off`); injected OFF/error tests verified adapter rejection and fallback. Optional-kernel rejection cannot repair a process whose own system runtime cannot execute.

The production checks were not weakened to make a test pass. CPU capability or execution-domain changes still require re-preparation or process restart; transparent hotplug is outside this prototype.

## Reproduction boundary

`bash examples/TritonSoftmax/reproduce.sh` performs a fresh AOT export and cross-build. The formal host-preparation A/B experiment instead used `prepare_host_optimization.sh` with a previously verified `BASE_ARTIFACT_DIR`, a new `ARTIFACT_DIR`, and `HOST_PYTHON` so the kernel and Buddy graph objects stayed identical. On the target, extract the bundle into a fresh directory and run `python3 verify_board.py --artifact-dir . --cpu <allowed-cpu>`. Timing is a separate step using `compare_host_modes.py`; the correctness counts above are not latency measurements. Machine-specific paths and complete transport logs remain outside the PR.
