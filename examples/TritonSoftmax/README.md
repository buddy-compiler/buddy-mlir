# Triton-RISCV Softmax through Buddy/RAX

This is the minimal proof of concept for [Issue #918](https://github.com/buddy-compiler/buddy-mlir/issues/918). The [interface RFC](../../docs/TritonKernelInterface.md) defines the ABI, metadata, memory ownership, dispatch, errors, and fallback. The implementation is reusable at the ABI/runtime boundary; the exported example is one fixed specialization.

## Supported profile and execution path

- Linux riscv64, little endian, LP64D, RVV 1.0, VLEN 256. The full required ISA comes from the generated `export.json` and is checked before invocation.
- Finite contiguous 2×4 FP32 input and output, element strides `[4,1]`, Softmax over the last dimension (`axis=1` or `-1`). Input/output must not overlap. Host memory, synchronous execution, serial example Session.
- Preparation checks the online CPU domain; execution may migrate within the verified compatible domain. Benchmarks pin one CPU only to reduce noise.
- The kernel is AOT compiled into a shared library. Executing it does not require Python or Triton JIT. The public ABI carries dynamic tensor metadata, but this graph/profile does not accept arbitrary shapes or layouts.

```text
RAX → actual buddy-cli → SoftmaxRunner → compiled Buddy computation
    → Buddy Runtime → adapter → Triton kernel → caller-owned output
                  └ no compatible optional kernel → Buddy-generated Softmax
```

The fixed input is `[[1,2,3,4],[0,0,0,0]]`. Expected output rows are approximately `[0.0320586,0.0871443,0.2368828,0.6439143]` and four values of `0.25`. `Run.cpp` also compares 1,000 deterministic finite inputs against the real Buddy-generated fallback with `atol=1e-6`, `rtol=1e-5`.

## Build prerequisites

Use the existing Buddy/LLVM build (including `buddy-opt`, `rax-pack`, `rax-inspect`, `buddy-llc`, the generated RAX header, and CLI11), a working Triton-RISCV Python environment, and a RISC-V Linux cross C++ toolchain/sysroot. The target board runs the resulting ELF files; it does not build LLVM or Buddy. The checked environment used Clang/LLVM 24.0.0git and LLD 19. A changed Triton backend requires rechecking the generated LLVM IR, object symbol, raw signature, and link dependencies rather than assuming the same internal ABI.

`CROSS_CXX` can be a cross compiler or a wrapper around LLVM Clang. An example wrapper is:

```bash
#!/usr/bin/env bash
set -euo pipefail
exec "${BUDDY_LLVM_BUILD_DIR:?}/bin/clang++" --target=riscv64-linux-gnu \
  --sysroot="${RISCV_SYSROOT:?}" --gcc-install-dir="${RISCV_GCC_INSTALL_DIR:?}" \
  --ld-path="${RISCV_LINKER:?}" "$@"
```

## Fresh AOT export and cross-build

From the Buddy repository root, point the variables below to existing tools and choose an empty artifact directory:

```bash
export BUDDY_BUILD_DIR="$PWD/build"
export LLVM_BUILD_DIR="$PWD/llvm/build"
export CROSS_CXX=/path/to/riscv64-cxx-wrapper
export TRITON_REPO=/path/to/triton-riscv
export TRITON_PYTHON=/path/to/triton-env/bin/python
export BUDDY_PYTHON=/path/to/buddy-env/bin/python
export HOST_PYTHON="$BUDDY_PYTHON"
export FLATBUFFERS_INCLUDE_DIR=/path/to/include
export ARTIFACT_DIR="$PWD/build/triton-softmax-new-run"

"$HOST_PYTHON" examples/TritonSoftmax/check_metadata.py
bash examples/TritonSoftmax/reproduce.sh
```

The script AOT exports the kernel, cross-compiles the adapter/Buddy graph/Runner/actual buddy-cli and fixtures, packages RAX, and validates metadata, target ELF files, and dependencies. It does not execute target code on Ubuntu. The expected build milestone is `PASS packaged ... RISC-V ELF artifacts; no target executed`; the result is `$ARTIFACT_DIR/test-bundle.tar.gz`. The script refuses to overwrite a nonempty artifact directory.

`export.py` checks the generated ABI and emits the object/metadata; `softmax_metadata.py` supplies the shared build-time profile. `Adapter.cpp` maps the public call to the actual 13-argument Triton entry; `SoftmaxSession.cpp` contains the compiled graph boundary and Buddy fallback; `SoftmaxRunner.cpp` binds RAX and selects candidates. Shared ABI, validation, and loading live in `runtime/include/buddy/runtime/core/ExternalKernel*.h` and `runtime/core/ExternalKernel.cpp`.

## Correctness on the target

Transfer the test bundle using an existing transport, extract it into a new target-side directory, and run there:

```bash
python3 verify_board.py --cpu <allowed-cpu>
```

The CPU number must belong to the actual allowed set. The verifier checks bundle hashes and a fresh extraction cache, then runs standalone invocation, the compiled Buddy/Triton comparison, 56 actual CLI/RAX variants, and 50 contract groups, including three host-preparation groups. Expected-error cases must match their diagnostics and leave output sentinels unchanged. A successful run prints `PASS clean replay`. Keep `board-summary.json`, `rax-results.json`, `contract-results.json`, and the command output. The target-side scripts require only Python's standard library, `taskset`, and the target's normal runtime libraries.

The latest board replay passed 1,000 finite Buddy comparisons (maximum measured absolute difference 0, `atol=1e-6`, `rtol=1e-5`), 56 variants and 50 contract groups. See the [correctness report](../../docs/TritonKernelHostPreparationResults.md). A dynamic C++ test image with vector `NEXT=OFF` trapped in dynamic startup before the adapter loaded; this is not recorded as a passing application scenario. A separate freestanding static probe verified the exec policy, while adapter injection tests verified rejection/fallback behavior.

## Optional overhead measurement

Run timing only after correctness succeeds. The formal library exports only the checked query function; a separate measurement library contains the raw-kernel entry. On the target, from the extracted bundle:

```bash
python3 benchmark.py run --cpu <allowed-cpu> --processes 3 --iterations 200000
python3 benchmark.py summarize --results benchmark-results/results.json \
  --output benchmark-results/summary.json
```

The strict/prepared comparison uses `compare_host_modes.py` and four alternating A/B/A/B groups. The [overhead report](../../docs/TritonKernelHostPreparationOverhead.md) gives the exact commands, distributions, cold costs, and measured platform. This 2×4 case performs no data-layout conversion; Buddy fallback's 32-byte return adaptation is included in its end-to-end time. The prepared Triton graph remains slightly slower than Buddy, and its first-session preparation is more expensive. Optional `perf` sampling did not have permission on the measured system; no system setting was changed.

## Failure and packaging boundaries

An inconsistent `export.json`, generated header, or object digest must be rebuilt from the same version; do not alter RAX to hide the mismatch. Missing/incompatible optional libraries fall back to Buddy for a valid graph call. A missing mandatory resource, corrupt payload, or invalid graph binding is an error. The target package does not promise arbitrary shapes, floating-point scalar types, nonzero workspaces, in-place execution, transparent CPU hotplug, or broader operator coverage.

Downstream CMake users can link `BuddyMLIR::buddy_runtime_external_kernel` from the build tree or `find_package(BuddyExternalKernel CONFIG REQUIRED)` after installation. The fixture at `tests/Runtime/core/Inputs/ExternalKernelCMake/` covers executable/shared consumers from build-tree, installed, and relocated configurations; all six target-side cases passed.
