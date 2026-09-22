# AME v0.5 cache contract and RA/Nanhu synchronization

This note distinguishes the supplied AME v0.5 documentation, the platform
developer's subsequent fence recommendation, and the actual linked code.
The developer's recommendation supplies a concrete sequence for a diagnostic
comparison; a separate cache API is not a prerequisite for that comparison.

## Evidence in the supplied documents

The source is
`references/AME v0.5 软件使用说明（2026-9-6版本）.pdf`, section 5,
“Cache 与内存所有权” (PDF page 6). It says:

* AME is a non-coherent master.
* `fence rw,rw` orders CPU/AME accesses but is not cache clean/invalidate.
* CPU/NH writes to A/B/C must be clean/flush-synchronized before an AME load.
* An AME store to O must be followed by CPU/NH invalidate before the CPU/NH
  reads O.
* Linux should use the E6 driver's `SYNC_MEM(CPU->E6)` and
  `SYNC_MEM(E6->CPU)` paths.
* Disabling the E6 D-cache through a CSR is only an isolation experiment, not
  a software ABI.

Section 6 (PDF pages 6--7) gives a small NH→E6→AME INT8 example. Its assembly
ends with `fence rw, rw`, but it does not define a general RA clean/invalidate
instruction, CSR number, or DMA API. The example therefore cannot override
the explicit cache rule in section 5.

The related hardware document,
`references/RAV0.5_FPGA_ALPHA_260908 硬件文档.pdf`, identifies the same
RAV0.5/AME integration and points to the section-6 software example, but adds
no RA cache-maintenance ABI.

## Read-only server cross-check

The RAV0.5 server tree was inspected without changing it. The example is:

`/home/hjuser/Desktop/RAV0.5_FPGA_ALPHA_260908/test/software/nh_e6_ame_gemm/nh_e6_ame_gemm.S`

It uses unused SYSCTRL boot-address registers as uncached NH/E6 handshake
locations, and deliberately avoids an NH read of O before the AME write. It
contains `fence rw,rw` around the handshake and the AME sequence, but no
`SYNC_MEM`, DMA sync call, RA `cbo.clean`/`cbo.flush`/`cbo.inval`, or cache CSR
sequence. This arrangement avoids some cache-sharing cases; it does not
exercise our repeatedly reused DDR console ring. No synchronization API was
identified in the inspected files. The earlier broad server search had capped
output and does not prove that no such implementation exists anywhere on the
server. No server files or SYSCTRL registers were changed during this check.

This is evidence that the supplied server example does not answer the missing
ABI question; it is not evidence that the platform is coherent.

## Developer clarification (2026-09-20)

The user relayed this concrete definition from the platform developers:

```c
static inline void ame_intrin_fence_rw_rw(void) {
    asm volatile("fence rw, rw" ::: "memory");
}
```

The two supplied screenshots under `references/fence/` show:

* `ame_fence.png`: `run_ame_group` calls this helper after
  `ame_intrin_msce32_acc0(accumulator, ...)` and before returning.
* `fence_after_kernel.png`: `run_packed_case` calls `fence()` after RVV
  forward/backward kernels and before checking their output with the CPU.

Neither screenshot contains the helper definitions or startup/cache
configuration. The first helper is identified by the definition supplied in
the message; the second helper's exact definition is not independently shown.
The first screenshot also mentions GEM5 and shows a packed B buffer. It is
evidence for fence placement, not a reason to import its matrix layout or old
instruction wrappers into the v0.5 implementation.

Our `nr_runtime.c::fence()` has the same assembly and `memory` clobber as the
provided definition. `volatile` keeps this assembly observable to the compiler;
the clobber prevents compiler memory reordering across it. Whether this
particular FPGA implements additional cache effects is still not established
by the screenshots. The general ISA ordering guarantee and section 5's
non-coherent-master warning remain distinct from platform-specific behavior.

## Current repository status and machine-code evidence

`examples/FPGA-BOSCAME/common/nr/nr_runtime.c` has NH-side `cbo.flush` and
`cbo.inval` helpers for the console/RX shared rings. The RA AME helpers in
`examples/FPGA-BOSCAME/common/nr/ame_sync.c` use fences around AME instructions
but do not perform a documented RA→AME clean or AME→RA invalidate. The
diagnostic hang-watch code deliberately does not add an unverified cache
instruction.

The existing target `matmul_1x1024x2048` already has the first screenshot's
AME-store/CPU-read fence. Its archived `kernel.nr.S` has MSCE32 at line 215,
`fence rw,rw` at 216, and the CPU accumulator load at 241. Its scalar output
copy at 269 happens later; there is no separate function-tail fence. The next
RVV consumer has fences around its memory instructions. The compiler-output
filter `tools/restrict_fpga_assembly.py` inserts fences before and after AME
and RVV memory operations, and `tools/check_nr_elf.py` checks their adjacency
in the final ELF.

The locally built diagnostic image
`model/build/hang-watch-20260920/blocking/image/qwen_model.elf` has SHA256
`8e40b9ccb4dbd83f69d5e37a792c1d9336e2785945ae34f5252754342830576c`:

```text
0x8017671a: 0330000f  fence rw,rw
0x8017671e: 02d52077  MSCE32
0x80176722: 0330000f  fence rw,rw
0x8017675c: 00032303  lw t1,0(t1)  # CPU accumulator read
```

These addresses apply only to that image. They prove fence presence, not its
hardware execution or the correctness of addresses at the time of a failure.

Our public `ame_fence()` is different from the supplied helper: it issues
`fence; ame_resync_state(); fence`. The middle call configures a 1x1 tile and
issues extra AME loads/MQMA/load/store. Previous `--profile-sync=fence`
experiments replaced only the profiler's extra synchronization; graph-final
`ame_fence()` remained. A profiler `returned` message also precedes that
profiler fence, so console blocking could prevent reaching it.

## Optional RA CBO diagnostic hook

Because the supplied documentation does not define a bare-metal RA
`SYNC_MEM`/CBO ABI, the common runtime now exposes two range functions,
`nr_ame_cache_clean()` and `nr_ame_cache_invalidate()`, that are no-ops by
default.  Defining `NR_RA_AME_CACHE_DIAGNOSTIC=1` when compiling
`common/nr/nr_runtime.c` makes them walk a configurable (default 64-byte) line
range with `cbo.flush` and `cbo.inval`, respectively, bracketed by `fence
rw,rw`.  This is a diagnostic comparison only: it is not the documented
`SYNC_MEM` path, has no evidence of support on the current RA, and can itself
trap as an unsupported instruction; `cbo.inval` can also discard dirty CPU
lines if the caller uses it before publishing a result.  A caller must pass
the complete contiguous bytes for each AME input/output; descriptor strides and
aliases must be handled by the caller.  The default production image does not
emit these instructions and is unchanged.

The local image builder now supports `--graph-sync=fence`, independently of
`--profile-sync`, to replace only the graph-final resync with the developer's
single fence. With silent `--hang-watch` and profiling disabled, the comparison
does not add per-kernel UART or profiler resync. Production defaults and kernel
objects remain unchanged. See [the local diagnostic procedure](../model/optimization/HANG-WATCH.md).

Neither adding a fence nor changing an instruction and observing one PASS
proves a particular hardware defect: timing, layout, cache state, and repeated
trials must be considered. These observations support controlled experiments
without requiring an invented RA cache instruction or a hardware configuration
change. No diagnostic image has been run as part of this clarification.
