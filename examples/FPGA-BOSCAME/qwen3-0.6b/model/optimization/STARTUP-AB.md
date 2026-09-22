# Startup hang investigation, 2026-09-22

## Evidence and limits

The original production image completed twice on FPGA5 (`run-51ef77a6f71b4e4f`,
`run-0617b5a11ec94e32`). The startup-prime image timed out once
(`run-96a031f6164e40a2`): prefill validation completed, then UART stopped after
`decode begin position=00000010`. This is not yet a repeated prime failure.
Host worker CPU time does not measure FPGA CPU activity or identify a hardware
wait. The last UART line does not identify a stopped PC. DDR load readbacks
validate initial loading, not memory contents at the later failure.

The failed ELF SHA256 is
`fed6d5bf1a3612e4e653ebb0e781ce27f2f30ef3d794bfc9fd66a6d6d763ad0d`, at
`build/console-fix-20260921/production-prime2/image/qwen_model.elf`.
Compared with production-control, only `model_main.o` differs among linked
objects. The additional guarded startup operation changes both execution and
layout. In particular, the adapters' BSS moves from `0x810177e0` to
`0x81017828` (+0x48), including `qwen_out_*` and RMSNorm scratch. AME sync
buffers and stack also move. The high workspace/weights/KV addresses agree.

Compiling the existing source-level startup variants does not isolate layout:
`model_main.o` text is 149368 bytes for prime, 149352 for layout, and 149360
for fence. A compiler barrier alone does not keep instructions or addresses
fixed. Use the exact-byte controls below before attributing an outcome to AME
state or buffer placement.

## First experiment: preserve the failed image's layout

Use `tools/prepare_startup_ab.py` with the exact failed prime ELF. It must reject
a different input hash, verify both startup calls and graph-final calls, and
change only these two four-byte instructions in a new ELF:

| Startup call PC | Original instruction | Layout control | Fence control |
| --- | --- | --- | --- |
| `0x80150c78` (prefill) | `jal ra,ame_fence` | 32-bit NOP | `fence rw,rw` |
| `0x80162412` (decode) | `jal ra,ame_fence` | 32-bit NOP | `fence rw,rw` |

The guard and primed flag remain. With an intact primed flag, only the prefill
startup action executes. The graph-final calls at `0x801558d6` and
`0x80167072` must remain unchanged. All other bytes, section metadata, symbols,
stack allocation, kernel code, buffers and addresses must match the prime ELF.
NOP/fence controls omit the helper's execution and therefore its transient
register/stack/cache effects; these remain part of the startup-action factor.
They do not isolate individual AME instructions within the helper.

1. Build and audit the fixed-layout NOP control. Check undefined symbols, byte
   differences, ELF/bin hashes and generated DDR plan before FPGA use.
2. Run it on FPGA5 with the original 28 layers, capacity 128, fixed prompt,
   tokenizer, weights, 16-token prefill and eight decodes. Retain the existing
   4 MiB append-only console with live drain and graph-final AME resync.
3. If it passes, build/audit/run the fixed-layout ordinary-fence control.
4. Rerun the exact original production-control image as a contemporaneous
   control. Rerun the exact prime image to establish whether the original
   failure recurs. Alternate the discriminating pair if outcomes differ.
5. If layout also stalls, prioritize image layout and runtime corruption;
   skipping startup AME was insufficient. If layout/fence pass and prime
   repeatedly stalls, narrow the startup helper, including AME state, memory
   traffic and timing. One trial per image only nominates a hypothesis.

Use one UART owner and the existing runner, only under the approved remote
root `/home/hjuser/Desktop/fpga-tester-ISCAS`. Do not change symlink targets,
other sessions, hardware configuration, or `examples/BuddyQwen3`. Keep existing
artifacts. No profiling, hang-watch, intermediate probe, workspace CBO, or
additional runtime logging in this experiment.

## Evidence for every run

Record raw and prepared image SHA256 separately (the prepared image has
64-byte padding), ELF SHA256, input hashes, command, run ID, original UART,
worker/UVHS logs, DDR readback results and remote readback paths. Record host
wall/CPU times with their process scope; use board cycle output for FPGA
compute time. Use the same 1800-second capture limit and 900-second startup
limit. A successful run requires all 27 logits/KV comparisons, the nine-token
trajectory, fixed text validation and `RA returned: PASS`.

For a timeout, retain the full configured capture and last progress timestamp;
do not label the last printed kernel as the hardware PC. Preserve the worker's
terminal result before launching another image. A disconnected SSH client
must resume the same worker, not reload the FPGA.

## Second experiment: locate the failing mechanism

Choose the next experiment from the first results:

* Layout-sensitive: compare adapter BSS alignment/placement, AME sync buffers,
  code and stack separately, using fixed linker placement and a full map diff.
  The adapter BSS +0x48 change is a concrete candidate, not an established bug.
  The NOP image also retains new primed-flag loads/stores. If the exact none
  control still passes, first consider an address-identical guard-bypass image:
  replace the two four-byte `lbu a0,0(s1)` at `0x80150c72`/`0x8016240c` with
  `addi a0,zero,1`. The existing branch then skips startup call and flag store
  on every invocation, with all addresses fixed. Validate the surrounding
  register liveness before using this diagnostic. It separates the group of
  guard memory accesses/branch timing from the other prime-layout differences;
  the implemented experiment and terminal result are recorded below.
* Startup-action-sensitive: keep the image fixed while reducing the helper's
  operations in coherent groups (configuration, operand loads/MQMA, accumulator
  load/store). Preserve required fences and avoid invalid instruction sequences.
  First consider patching only the prefill or only the decode startup call:
  the decode call should be skipped by the primed guard. A change from patching
  that supposedly unreachable call would require investigating flag corruption
  or stale visibility before attributing failure to the initial prime.
* Mixed/unreproduced: interleave exact images with identical reset/load flow;
  record platform state rather than assuming deterministic reproduction.

Before adding software instrumentation, inspect existing documented access to
RA PC and AME/bus state. If available, take repeated PC/state samples during a
stall to distinguish AME wait, memory wait, software loop and corrupted
control flow. Otherwise introduce one bounded observer at a time and retain
an uninstrumented control, since earlier instrumentation moved failure points.
Investigate the bare-metal SYNC_MEM contract independently; do not invent a
cache ABI or treat the existing RA CBO diagnostic as a fix.

Root-cause closure requires a repeated discriminating experiment plus a
specific mechanism supported by PC/state or a minimal reproducer. A passing
variant alone is a workaround observation, not that closure.

## Additional static checks

Adapter buffers satisfy their declared 4-byte alignment in both original
images. The shift changes their offset in a 64-byte line from 32 to 40, but
the accepted RMSNorm kernel already performs RVV copies from an internal
stack buffer at offset 40 in a line. There is no demonstrated alignment ABI
violation. Adjacent adapter arrays, scratch and `heap_cursor` share boundary
cachelines in both images; the split changes from 32/32 to 40/24. This is a
testable CPU/RVV memory-visibility hypothesis, not proof of corruption.

The allowed platform scripts expose `readback_reg` examples for UART/DDR
activity, but no validated RA PC or AME outstanding path. Probe insertion is
disabled in the inspected frontend/backend scripts. Do not assume an internal
signal can be sampled from the installed bitstream without checking support.

Static stack inspection found identical `run_prefill`/`run_decode` fixed
frames of 96288 bytes in control and prime. Including the large C-interface
argument expansion (165248 bytes), graph frames and the largest compiled
kernel stack (about 37104 bytes), the inspected call structure stays below
roughly 323 KiB of the 1 MiB RA stack. Temporary graph descriptors use paired
stack-save/restore operations. This is not runtime SP evidence, but there is
no ordinary stack-capacity exhaustion demonstrated by the generated code.

## Initial artifact

Fixed-layout NOP control: `build/startup-ab-20260922/layout/`.

* ELF: `c7ebf6a2b54bb0feef13f2d1cc0d1a7ce15fb813bc9043c6df6aaf2743d2401c`
* Raw BIN: `65640cf046b085550b7adcd23323338d044d4ae7f6a409d2059f2a52c2c3c61c`
* Prepared BIN: `a45b7ae0bf8c07ec7f53d729275cd9c5b494b0f049d87fa821370c81db3e2876`

ELF audit passes with no undefined symbols. Seven bytes differ within the two
four-byte patch sites; all other bytes and symbol/section tables are identical.
The tool's four local validation tests pass. Board outcome is recorded separately.

Run `run-616f460610fe462c` is the first fixed-layout NOP trial. It completed
prefill and decode positions 16, 17 and 18 (12/27 full-tensor checks), then
stopped producing model UART after `decode begin position=00000013`. The log
remained 11910 bytes at worker elapsed 489 through 1780 seconds. The run then
reached its 1800-second capture timeout with no completion marker. One trailing
NUL arrived during session shutdown (final length 11911 bytes); preserve it in
the raw evidence and do not treat it as model progress. All three load-time DDR
readbacks match. Host orchestration wall time was 2008.34 seconds, user CPU
4.04 seconds and system CPU 1.54 seconds.

The complete derived-image archive is
[`layout-run-616f460610fe462c`](../validation/board/startup-ab-20260922/layout-run-616f460610fe462c/verification.json),
marked `NOT_ACCEPTED`. The startup AME action is therefore not necessary for
every observed stall. Layout, retained primed-guard accesses, transient timing
and platform variability are still confounded. The original prime failure
and this later decode-19 timeout are not proof of the same mechanism.

The subsequent exact original none run, `run-1f8b9ec0c3894135`, completed
all eight decode steps, 27 full logits/KV comparisons, fixed-text validation,
and `RA returned: PASS`. All three load-time DDR readbacks match. Its independent
archive is [`none-run-1f8b9ec0c3894135`](../validation/board/startup-ab-20260922/none-run-1f8b9ec0c3894135/verification.json),
with status `MODEL_RUN_NUMERIC_PASS`. Local host orchestration wall/user/system
times were 1267.53/4.10/1.35 seconds; these are not RA execution times.
This supports investigating the retained startup guard and changed low-address
layout, but a single trial of the NOP image does not establish repeatability.

The hardware team's K=64 reissue patch targets `LowerQwenW8A8ToBOSCAME`.
The current Triton archive instead uses `LowerLinalgToBOSCAME`; all 11 AME
matmul templates already reissue K after loading C and before their first A
load. The remaining 35 kernels do not issue AME. Applying the supplied patch
to the unused lowering would not test a changed image. One remaining ordering
question is K-before-INT8-type versus the current INT8-type-before-K order;
that stronger hardware requirement has not been established.

Prioritize the exact original none control; defer the ordinary-fence experiment
because the fixed-layout NOP image did not complete. Original none metadata
and orchestration logs live in `build/startup-ab-20260922/none/`.

The helper's static register/tile audit found no demonstrated GPR ABI defect.
The current synthetic sequence executes MQMA before loading Acc0 with MLCE;
its computed value is then overwritten. The documentation does not establish
that using the old Acc0 here causes a hang. The extra generic msettype values
also lack a complete platform state contract. Crucially, graph-final resync
executes this same helper after prefill in both images, so “prime leaves INT32
mode” alone does not explain a difference after that common operation.

## Address-identical guard bypass

`prepare_startup_ab.py --variant guard-bypass` now implements the proposed
guard experiment. It verifies the original source hash and both guard windows,
then replaces only `lbu a0,0(s1)` at `0x80150c72` and `0x8016240c` with the
32-bit `addi a0,zero,1`. The unchanged `c.bnez` takes the existing skip edge;
the startup call and primed store are not executed. Each skip target begins by
overwriting `a0`, so the forced value is used only as the branch predicate.
Both startup-call instructions remain present, as do both graph-final calls.
There are six changed bytes in the ELF; every other byte and address is intact.

Artifact directory: `build/startup-ab-20260922/guard-bypass/`.

* ELF: `eb318dc739a10017355c082a265b7397e1cad54da90433494bed98f8a24050f0`
* Raw BIN: `b4c12987c786526797c3153045515513b3ae56984b828aec204a5b0a32350d3a`
* Prepared BIN: `70e004ccbfd58a9b2e16657a995f9c69151f81677bca12259df72dfda5d6b457`

The ELF audit passes with no undefined symbols. Eight local startup-variant
tests pass, including verification that the original layout/fence variants
still produce their original byte patches and invariants. FPGA5 run
`run-e92fa9f93bb841fd` completed with the worker's capture-timeout error, using
the same workload, 1800-second capture and 900-second startup timeout. Prefill
and decode 16/17/18 passed (12/27 full-tensor checks), then model UART stopped
after `decode begin position=00000013 input_token=00000C4A`. UART remained
11910 bytes through worker elapsed 1779 seconds. During shutdown another 65
non-model bytes arrived; the complete 11975-byte raw log is retained unchanged.
All three load-time DDR readbacks match, including independent remote hashes
recorded in the build directory's `remote-readback-hashes.json`. Host
wall/user/system times were 1994.74/4.31/1.63 seconds.

The independent archive is
[`guard-bypass-run-e92fa9f93bb841fd`](../validation/board/startup-ab-20260922/guard-bypass-run-e92fa9f93bb841fd/verification.json),
with status `NOT_ACCEPTED`. Like the NOP control, this image stopped producing
model UART at decode 19. The skipped startup helper and guard load/store are
not necessary for this observation. This does not prove that the earlier
prime decode-16 failure has the same mechanism; graph-final helper calls,
primed BSS storage and CRT clearing remain present.

Compared with the NOP startup control, this removes guard loads/flag stores
and changes branch behavior and timing, not just memory visibility. A PASS
would nominate that group of effects, not prove flag corruption. A stall
would show that neither the startup helper nor guard memory accesses are
necessary for the new observation; repeatability and the actual stopped PC
would still need establishing.

## RA SP-only control

`build/startup-ab-20260922/prepare_stack_control.py` derives `stack-minus64`
from the exact guard-bypass ELF. Only the RA normal and trap entry ADDI
immediates change, at `0x8000005c` and `0x800000ec`. Actual initial SP moves
from `0x82108c80` to `0x82108c40`, the original none image's SP value. This is
two changed bytes; all other ELF bytes and symbol addresses are unchanged.
The `__ra_stack_top` symbol still denotes the reserved region's original top,
while actual usable RA stack capacity is 1048512 bytes (64 bytes less).
The stack base, heap start, adapter BSS and AME sync buffers do not move.

* ELF: `052afe677279118a447de7cff2d20dbff29b0d861b7213dac37274edcd6f8972`
* Raw BIN: `8ce6a116dfd720fc7936bb00e0dcab95ad22f4f13691b1ef869ebcfe71a4a672`
* Prepared BIN: `cf0a5fbaaad9cb84e9043c9d138c24c3910407ab3a457ae8cb2337608cbdd282`

ELF audit, undefined-symbol check, reverse-patch identity, unchanged guard and
completion instructions, and two-hop derivation validation pass. Independent
archive tooling checks prime -> guard -> SP and retains both manifests;
12 invalid identity/derivation cases are rejected. No original image.json is
rewritten to describe the derived image.

FPGA5 run `run-52d9353394064ff4` was launched after the guard worker's terminal
result and local exit, with the same capture/startup limits. Its result is
pending. A PASS would nominate stack-address/cache-mapping/timing effects,
not prove a cache or stack root cause. If the pair separates, interleave
parent and SP trials. If it does not, move to a minimal boundary observer
instead of trying arbitrary address offsets.

## Prepared fixed-address boundary observer (not yet run)

The offline script `build/startup-ab-20260922/prepare_boundary_observer.py`
accepts only the audited guard or SP parent hashes. It redirects three decode
calls through nearby veneers and wrappers in existing zero-filled ELF padding:

| Marker | Original call PC | Positive evidence |
| --- | --- | --- |
| `[boundary] B` | `0x8016706e` | About to call the decode graph |
| `[boundary] G` | `0x80167072` | Graph returned; about to call graph-final resync |
| `[boundary] S` | `0x80167076` | Graph-final resync returned; collection is still ahead |

No original code/data/symbol VMA moves and no BSS is added. Added code is
explicitly covered by expanded executable sections and boot LOAD extent,
not hidden outside ISA-audited sections. It uses the existing
`write_serial(char)` append-only path, saves all a/t registers, restores the
original SP before entering the original callee, and leaves stack arguments
intact. The callee RA points into the wrapper, and the wrapper reconstructs
the original return PC afterward without destroying a0/a1 return values.

Thus the observer still perturbs memory traffic (a temporary 128-byte stack
frame and console writes/fences), callee RA/its saved stack value, instruction
fetch and timing. Missing markers cannot prove a boundary was not reached:
console publication freshness is not newly guaranteed. There is no new cache
ABI, no RA CBO, and no independent collect/UART liveness observation.

Independent checks passed: source identity, all nonpatch bytes unchanged,
ELF-to-BIN section mapping, legal LOAD extents, undefined-symbol check, full
ISA audit (536873 instructions, 131 AME, 408 RVV), and actual-machine-code
ABI simulation for three hooks at two SP values. The guard/stack observer
ELFs and BINs differ only at the original two SP-immediate bytes, so the
observer is identical if a paired follow-up is needed.

The prepared guard experiment lives in `build/startup-ab-20260922/guard-observer/`:

* ELF: `96816664237d60c94d0be75d91675b0804ac45f5cc3e8152fd90b65e3b6c297a`
* Raw BIN: `ba1f40d922c4cfb904cc1b6e7736b22d201ce72c785da48e90ac6f57bfa5b574`
* Prepared BIN: `ff4a1a04fb4a51e058c7f633ed45e1c415f7479d19ebfabf25f998e132f5e6d4`

This is not a board result. The earlier `guard-observer-local1` directory is
a retained failed offline link attempt and must not be deployed.
