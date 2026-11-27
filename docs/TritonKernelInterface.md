# Interface for invoking Triton-RISCV kernels from Buddy/RAX

This RFC addresses [Issue #918](https://github.com/buddy-compiler/buddy-mlir/issues/918). It defines a reusable, optional kernel boundary and validates one AOT-compiled Softmax specialization. The design and the implementation scope differ: the versioned ABI carries runtime tensor metadata, while this prototype accepts only a contiguous 2×4 FP32 input with finite values. The target is Linux riscv64, little endian, LP64D, RVV 1.0, and VLEN 256. Execution is synchronous on host memory.

## 1. Integration boundary

| Component | Responsibility |
| --- | --- |
| Buddy compiler | Generate the Buddy computation and its original fallback; emit an explicit optional dispatch point and pass tensor/scalar operands. The prototype does not add an automatic operator replacement pass. |
| RAX package | Carry the computation, mandatory fallback, optional kernel code object, compatibility metadata, argument bindings, and a relative file URI or embedded payload. It never stores live pointers. |
| Buddy Runtime | Prepare and retain candidate libraries, validate the call and target, invoke a selected kernel synchronously, report status/path, and run the Buddy fallback when a compatible optional kernel is absent or execution fails. |
| Triton-RISCV compiler | Perform specialization, lowering, and RISC-V object generation through its existing backend. Buddy does not duplicate this logic. |
| Compiled Triton kernel | Execute its actual generated entry and write the borrowed output. It owns and releases its internal temporary allocations. |
| Export-side adapter | Convert the stable Buddy tensor call to the generated kernel's concrete argument list, pointer descriptors, and program grid. It must be rebuilt if the generated ABI changes. |

The kernel is **AOT compiled and delivered as a precompiled shared library**. Runtime execution does not JIT compile with Triton or require Python. `examples/TritonSoftmax/export.py` calls the existing `triton.compile(ASTSource(...))` backend. One specialization maps to one optional RAX code object and one shared library; multiple specializations can be represented by multiple code objects.

The reusable parts are the ABI, packaging, loading, validation, selection, diagnostics, and fallback. Fixed shape, axis, grid, and generated function signature belong to this Softmax profile, not to a model name. Further operators and automatic selection are follow-up work.

## 2. Kernel ABI

`runtime/include/buddy/runtime/core/ExternalKernelABI.h` is authoritative. V1 uses naturally aligned C POD structs under the native 64-bit Linux calling convention. C++ containers, exceptions, allocator ownership, and live pointers never cross the package boundary. A call borrows all memory until synchronous return; no tensor copy is needed to cross the Buddy/adapter boundary.

| Item | V1 convention |
| --- | --- |
| Tensor pointer | `input.data` and `output.data` point to the first logical element, including any slice offset; at least 4-byte aligned for FP32. Input is read-only. |
| Shape | `rank` plus a borrowed `int64_t shape[rank]`; dimensions are positive. The current graph requires `[2,4]`. |
| Strides | Borrowed `int64_t strides[rank]` measured in elements. The current graph requires `[4,1]`. |
| Data type | `BUDDY_KERNEL_F32=1`; RAX dtype is mapped by meaning, not cast numerically. |
| Layout | `BUDDY_KERNEL_ROW_MAJOR=1`; `Any`, column major, tiled, and opaque layouts cannot impersonate this profile. |
| Scalars | Borrowed `int64_t` array; the sole scalar is axis `1` or `-1`. The RAX scalar representation has no general type tag, so this prototype accepts only the writer's i64 binding. |
| Workspace | Caller-owned pointer and byte count, valid through the call. This profile requires `nullptr/0`; the field does not imply nonzero workspace support. |
| Result | The kernel writes the caller-owned output and returns a fixed-width `int32_t` status; no allocated result tensor crosses the ABI. |
| Status | `OK=0`, `UNSUPPORTED=1`, `INVALID=2`, `FAILED=3`. Runtime also reports the actual path and original external status. |

`byteSize` is the accessible span from `data`; it must cover `rows*cols*4`. Validation checks overflow, pointer alignment, tensor overlap, reserved fields, ABI version, and exact struct size. Tensor, call, and descriptor sizes are 48, 144, and 80 bytes on the supported 64-bit ABI. Dynamic shape/stride pointers avoid embedding or copying metadata in the fixed struct. The runtime still checks each call because its tensor contents and metadata can change between invocations.

Malformed calls return `INVALID` without invoking either path or changing the output. A valid call outside the specialization or an incompatible host returns `UNSUPPORTED` and leaves the output untouched. `FAILED` leaves the external output undefined; a successful Buddy fallback must overwrite it completely. Input remains unchanged in every case. Invalid graph bindings do not silently become default arguments or a fallback request. These rules are shared by Runtime and direct adapter invocation through `ExternalKernelValidation.h`.

### Generated Triton entry behind the adapter

The exported object has the following actual signature; it is deliberately separate from the public Buddy ABI:

```c
void softmax_kernel(
    int64_t output_rank, void *output_descriptor,
    int64_t input_rank, void *input_descriptor,
    int32_t input_row_stride, int32_t output_row_stride, int32_t n_cols,
    int32_t grid_x, int32_t grid_y, int32_t grid_z,
    int32_t program_x, int32_t program_y, int32_t program_z);
```

Each pointer descriptor is `{allocated, aligned, offset}` (24 bytes on this target); `rank=0` denotes a Triton pointer argument rather than the logical tensor rank. Output precedes input. The three runtime integers are `4,4,4`; `BLOCK_SIZE=4` is compiled in. The adapter calls two programs with grid `(2,1,1)`, program IDs `(0,0,0)` and `(1,0,0)`. The RAX host-wrapper grid `(1,1,1)` denotes one synchronous wrapper call and must not be confused with the internal Triton row grid. Changes to Triton `compiler.py` or `driver.py` require rechecking the LLVM IR, ELF entry, argument order, and unresolved symbols. The current object requires `expf`, `malloc`, `free`, and `memcpy`, resolved by libc/libm.

## 3. Kernel metadata and RAX packaging

The implementation reuses `runtime/rax/rax.fbs` without changing the schema. The Buddy computation and fallback remain mandatory resources. The external library is an `optional=true` host shared-library code object and is not a mandatory `DT_NEEDED` dependency of the Runner. `runner_library` uses the existing InferenceRunner plugin ABI; it is separate from the kernel query ABI.

| Metadata | Current representation |
| --- | --- |
| Identity and operation | Unique nonzero `CodeObject.id`, diagnostic `name`, `kernel_attrs.operation="softmax"`. |
| Compatibility | `dtype="f32"`, `shape="2,4"`, `strides="4,1"`, `axis="1"`, `layout="row_major"`; RAX buffer types must agree. |
| Target ISA | Canonical `required_isa`, `required_rvv_version="1.0"`, `rvv_vlen_bits="256"`, `required_matrix_features="none"`. The full compile requirement includes `i,m,a,f,d,c,v,zicsr,zifencei,zfh,zvfh,zba,zbb,zvfbfmin` plus deployment tokens. |
| Entry and location | `entry_symbol="buddy_external_kernel_query_v1"`, `kernel_abi="buddy.external.v1"`, `profile="softmax.f32.2x4.rvv1.vlen256.v1"`, and `file:` or existing embedded `payload:` URI. |
| Provenance | `export.json` records backend revision/hash, target options, object digest, generated signature/grid, and Triton version. The object hash is provenance, not a plugin signature. |

`softmax_metadata.py` is the build-time source for the profile. `export.py` emits the object, `export.json`, and `SoftmaxTarget.h`. The adapter includes the generated header; `run_rax.py prepare` validates the export and derives RAX attributes from the same record. The packer and inspector preserve and display these attributes. Bad-package fixtures are deliberately mutated only after the baseline record is validated.

The runtime parses ISA tokens case-insensitively and without order dependence, checks that RAX requirements are covered by the library descriptor, and separately validates actual host capability. Metadata strings alone do not prove CPU support. Query, constructors, and capability checks must use the baseline ISA before any RVV instruction executes.

Optional file resources resolve relative to the RAX package. Existing embedded payload extraction and cache are reused. For executable payloads, a cache hit is compared byte-for-byte with package content before reuse, preventing a stale same-size library from being run. Non-code resource caching is unchanged. This is consistency checking for trusted packages, not signature authentication or sandboxing of native plugins. The prototype assumes serial preparation in one process; concurrent extraction needs a later publication/locking design.

### Host compatibility preparation

Preparation checks the complete online CPU set against accurately parsed CPU ISA records and uses a temporary worker thread to inspect VLEN on every online CPU. It does not change the caller's affinity. If the set cannot be covered or changes during preparation, the optional candidate is rejected. The target must provide the full V extension and exact VLEN 256; matrix features are not required. After a compatible domain is established, the application thread may migrate inside that domain.

Each actual invocation thread checks its current Linux vector permission on first use and caches only a confirmed `ON` result in thread-local state. `OFF`, unknown, and errors reject this invocation and remain retryable. Linux's current `ON` state cannot be switched to `OFF` without exec; the adapter does not set an application's vector policy. Per-call shape/stride/pointer/alias validation remains. The default steady path does not repeat affinity or vector-policy syscalls. CPU capabilities or execution-domain changes require re-preparation or process restart; transparent hotplug is outside this profile. Strict per-call host checks remain only for diagnostic comparison. See the [Linux RISC-V vector documentation](https://docs.kernel.org/arch/riscv/vector.html).

## 4. Runtime dispatch and error handling

| Step | Rule |
| --- | --- |
| Lookup | Try an explicitly referenced candidate first, then other optional candidates for the same operation in package order. The first compatible candidate wins. |
| Load | During session preparation, use `dlopen(RTLD_NOW|RTLD_LOCAL)`, resolve the fixed query symbol, and check ABI, target, descriptor, ISA/profile, and host capability. Keep the library loaded through every synchronous call. |
| Prepare arguments | Validate RAX function/dispatch/buffer/scalar bindings and the original call; pass borrowed memrefs and axis through the compiled Buddy graph, then convert them in the adapter. |
| Ownership and sync | Caller owns tensor/metadata/workspace; Session owns optional library handle; plugin owns static descriptor and internal temporary allocations. Return means execution is complete. The example Session is serial. |
| Fallback | Missing or incompatible candidates continue lookup. If none is usable, or the chosen kernel returns non-OK, execute the existing Buddy-generated implementation and fully write the result. Do not try a second external candidate after an invoke has started writing. |
| Errors | Preserve candidate-rejection diagnostics, final status, actual path, and original external status. A malformed package or invalid graph call is an error, not an optional-kernel miss. |

A missing optional library, invalid query, or incompatible target does not make the mandatory Buddy computation unloadable. Corrupted FlatBuffer/payload, invalid explicit references, and invalid argument bindings fail clearly. A valid call with no compatible kernel returns `OK/Fallback` when Buddy succeeds. A kernel failure followed by successful fallback also returns `OK/Fallback` while retaining the external error. If fallback fails, output is unusable. SIGILL, process crashes, and memory corruption cannot be recovered as ordinary C ABI errors; compatibility must be checked before invocation.

## 5. Proof of concept and correctness

The actual path is `RAX → buddy-cli/InferenceRunner → compiled Buddy computation → Session/Runtime → adapter → Triton kernel → caller-owned result`. The fallback is generated from `torch.softmax(dim=-1)` by Buddy's existing lowering, not a hand-written C++ replacement. `computation.mlir` has an explicit `(input memref, output memref, axis i64, session pointer) → i32` boundary. `SoftmaxSession::compute` validates the original call before entering that fixed-shape graph, so invalid metadata is not reconstructed as a legal default.

The fixed input is `[[1,2,3,4],[0,0,0,0]]`; expected rows are approximately `[0.0320586,0.0871443,0.2368828,0.6439143]` and `[0.25,0.25,0.25,0.25]`. The board comparison used `abs(Triton-Buddy) <= 1e-6 + 1e-5*abs(Buddy)`, 1,000 deterministic finite inputs, an auxiliary double-precision oracle, and explicit path/status/fallback assertions. The measured maximum absolute difference against Buddy was 0 on those samples; the oracle's maximum absolute error was 7.2829e-8. These samples do not prove universal bit identity. The latest board replay passed 56 RAX variants and 50 contract groups (47 original groups plus three host-preparation groups), including genuine external execution and genuine Buddy fallback. Forty-two graph/runtime invalid cases and 43 direct-adapter invalid cases check unchanged output sentinels/canaries.

Six cross-linked CMake consumer cases passed on the board: build tree, installed leaf package, and relocated main package, each with an executable and a shared-library consumer. Their fallback callback tests the exported runtime contract; it is not substituted for the separate 1,000-case original-Buddy comparison.

The Ubuntu `check-buddy` run on the selected PR base discovered 253 example and 501 main-suite tests. Examples had 238 PASS and 15 UNSUPPORTED; the main suite had 497 PASS and 4 UNSUPPORTED, with no failures. Missing local Python optional dependencies were supplied from an isolated `build/` directory without changing the Conda environment; the previously failing five model-import tests also passed in a focused rerun. Native Ubuntu tests do not replace the RISC-V execution evidence.

## 6. Integration overhead

Measurements use the same cross-compiled object, fixed input, one CPU and one operator thread. Four alternating strict/prepared/strict/prepared groups each used three independent processes, ten rounds per process, and 200,000 calls per round: 120 steady batches and 840 cold/preparation samples. The paired dispatch estimate is computed within each round as full Buddy graph latency minus raw Triton path latency using the same library and object. It includes graph entry, adaptation, validation, and dispatch, so it is not a pure function-pointer-call cost.

| Metric (median) | Strict host checks | Prepared host checks |
| --- | ---: | ---: |
| Complete-graph minus raw-kernel dispatch increment | 1,394.386 ns | 173.392 ns |
| Data-layout conversion | N/A | N/A |
| End-to-end Triton graph | 2,135.017 ns | 913.263 ns |
| Same-window Buddy graph | 874.560 ns | 875.176 ns |
| First Session preparation | 6.040 ms | 11.940 ms |

The input is already contiguous row-major and borrowed directly, so no data-layout conversion occurs; N/A is not a measured zero. The kernel still performs internal staging/copy, and the Buddy fallback performs a 32-byte return adaptation copy; those costs are included in the corresponding end-to-end paths. Prepared host checking cuts the graph's steady latency relative to strict checking, but the tiny Triton graph remains about 4.35% slower than Buddy and cold preparation is more expensive. No automatic profitability decision or model-specific replacement policy is added. See `docs/TritonKernelHostPreparationOverhead.md` and its structured raw data for distributions and reproduction.

## Reproduction

Set the existing Buddy/LLVM and Triton-RISCV toolchain variables described in `examples/TritonSoftmax/README.md`, choose a new `ARTIFACT_DIR`, and run `bash examples/TritonSoftmax/reproduce.sh` on Ubuntu. This re-exports the kernel, cross-compiles the target artifacts, builds the RAX package, and checks ELF/metadata without running RISC-V code. Transfer the generated test bundle to a fresh directory on the compatible RISC-V target, then run `python3 verify_board.py --cpu <allowed-cpu>`. Benchmarking is separate and runs only after correctness. The prior A/B host-preparation experiment reused the same validated kernel/graph objects via `prepare_host_optimization.sh` so that the compared modes differed only in adapter host checking; its reused-object build is not the fresh-export reproduction path.

CMake consumers can link `BuddyMLIR::buddy_runtime_external_kernel` from the build tree or find the installed `BuddyExternalKernel` package. The static library is PIC and exports its include path, C++17 requirement, and `dl` link dependency. The consumer fixture is `tests/Runtime/core/Inputs/ExternalKernelCMake/`.
