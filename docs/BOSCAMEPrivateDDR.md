# BOSCAME private DDR frames

AME endpoints have four distinct proven sources: external entry arguments,
invocation-private DDR frames, fixed SPM windows, and unknown storage. An ordinary
aligned allocation is a candidate for a private frame; it is not permission to
issue an AME load/store against the CPU stack. The shared classification and
static ownership/bounds helpers live in Dialect/BOSCAME/Endpoint.h.

Default V2 emission collects private roots used by native AME tile accesses or
SPM boundary transfers. It acquires one frame at function entry and replaces
each root by a distinct aligned slice, retaining shape, descriptor offset
and physical strides. CPU-only DDR workspaces keep their ordinary allocation.
The function frame uses the sum of rounded slice sizes, with no speculative
lifetime overlap. Unknown/unbounded storage, pointer escape, unsupported memory
users and parallel/unknown regions cannot acquire direct access. The plan
verifier independently checks the same roots and the aggregate payload budget.

The allocation operation `bosc_ame.ddr.frame` returns a static `memref<Nxi8>` with
a positive, 64-byte-aligned payload of at most 262016 bytes. Only constants may
precede it in its single-block function. Exactly one `bosc_ame.ddr.release` must
occur immediately before return. Descriptor aliases remain local to serial
structured regions. Every descriptor alias, including a CPU-only view, must
have a statically proved bound inside the actual frame and preserve DDR memory
space. Tile endpoints must also prove alignment. Generic `memref.dealloc` may
not free a managed frame or any of its aliases. Neither `bosc_ame.direct_ddr`
nor `plan.buffer level="ddr"` establishes
these facts. An ordinary alloca with a forged attribute remains illegal.

Lowering builds a standard memref descriptor from the internal runtime call
`bosc_ame_private_ddr_acquire(N)` and releases it with
`bosc_ame_private_ddr_release(address,N)` after all AME values are dead. Existing
ordered AME load/store intrinsics decode this descriptor without adding a
second offset. Matrix boundary copies use their existing exact ACC32 geometry;
DMA and RVV copies keep their alignment, bounds and completion rules.
An actual one-physical-row view may use a rounded legal descriptor pitch because
no next-row address is accessed. Multirow and padded views retain strict pitch
checks.

The NH/E6 runtime requires linker symbols `__bosc_ame_private_ddr_start/end` and
`__nh_e6_ddr_window_start/end`. The private interval must be a separately reserved,
64-byte-aligned NOLOAD region inside the existing PMP operand window and
disjoint from user operands, image and stacks. Missing/invalid symbols trap when
a frame is requested. A 256 KiB arena holds a maximum 262016-byte payload plus
a 64-byte header/guard and a 64-byte trailing guard. Nested frames occupy
independent LIFO slices; capacity is checked across calls. Another hart,
concurrent allocator entry, malformed/double release or corrupted guards trap
through the private DDR failure hook. No singleton buffer is reused while a
prior invocation is active.

The default runtime allocator uses atomic busy/owner operations. The delivered
single-executor NH/E6 image explicitly compiles only its DMA runtime object with
`BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1`. Undefined or zero retains the generic
atomic implementation; other values are rejected. This target branch first
requires the observed E6 hart ID 0 and an actual SP inside the valid E6 linker
stack. Both NH/E6 stack ranges must be aligned, nonempty and disjoint; missing
symbols or an NH/outside-stack caller are rejected before allocator metadata
changes. A hart number alone does not distinguish the two processors.

Each target acquire/release saves and clears `mstatus.MIE`, then checks/sets a
volatile busy guard and checks owner under that same guard. Successful leave
copies the saved bit locally, finishes metadata and fences, clears busy, and
restores only the saved MIE bit. This bounded region covers frame metadata and
guards, not generated kernel execution or the lifetime of an allocated frame.
Logical nested LIFO calls remain supported. Reentry cannot overwrite an outer
operation's restore token. Failure hooks remain terminal; recovery through a
returning hook or target longjmp is unsupported. Resumable interrupt, NMI or
monitor handlers must not call this allocator. The installed E6 `mtvec` handler
halts; source inspection does not establish implementation-specific NMI routing.

ViT's `board.mk` selects this contract for its NH/E6 `dma.o` and records the
effective flag in the board input identity. Set
`BOARD_PRIVATE_DDR_NR_SINGLE_E6=0` to build its generic branch. The NH/E6
calibration builder records the same DMA-only selection. Other runtimes and
generic RISC-V targets do not select this branch automatically.

Private frames share the external operand region's visibility policy. CPU
stores use write-through policy, and CPU reads after AME writes use the delivered
uncached-load policy. Emission activates `bosc_ame.cpu_ddr_policy` for managed
frames and for mixed V2 functions containing both AME and scalar-DDR CPU work,
including functions whose AME buffers are entirely in SPM. The marker is a
module attribute: final lowering inserts policy transitions on CPU scalar/RVV
DDR accesses throughout that module, including CPU sibling functions and
ordinary stack accesses represented in DDR memory space. Those transitions
therefore also contribute CSR/fence overhead outside the private-frame path.
A standalone pure-CPU module does not acquire the marker from this rule.
Final lowering inserts the established read/write helpers and
rechecks matrix liveness after inserting calls. For the explicit `nr-fpga`
target, compiler-generated transitions then become side-effecting inline
`fence iorw, iorw` and `csrw 0xbc0` instructions with a memory clobber. The read
value is `0x0f0f000f000f000f`; the write value is `0x0f0f0f0f0f0f0f0f`.
This preserves live RVV values across the transition instead of introducing
an ABI call that spills them through unsupported vector CSR instructions.
Only marked calls to verified external `() -> ()` reserved helper declarations
are rewritten. Unmarked user calls and other targets keep their runtime ABI.
Compiler-generated NR transitions implement the delivered policy intrinsically;
overriding the runtime helper implementation or its macros does not alter
these instructions. Frame lifecycle and DMA helpers remain ordinary calls.
A fence alone is not treated as
cache invalidation. The ordinary stack remains cached and is not promoted to a
native AME destination. Frame acquisition/release execute outside matrix live
ranges.

The buffer simulator keeps external accesses in `Machine.accesses` and tracks
private accesses separately in `frame_accesses` and native AME spans in
`ame_frame_accesses`. `frame_log` and `private_frames` record bounds, guards and
release. Views preserve provenance independently of SPM bank identity and names.
Private storage starts with configurable nonzero poison (`private_poison`,
default 0xa5); frame guards use 0xd3. Runtime host tests cover nested/repeated
calls, exact capacity, bad windows/lengths, hart ownership, concurrency, release
order and corrupted guards. The target host model executes the same production
branch with both incoming MIE values, wrong/missing/overlapping stack bounds,
other harts, owner mismatch, synchronous reentry, restore-token preservation
and unrelated status-bit preservation. The compiler/object and simulator test
also checks that target runtime objects contain the intended CSR operations
and no AMOs, while generic objects keep their atomic operations. Its Machine
simulator exercises private A/B/P load/store, nonzero offsets, repeated calls
and mixed SPM transfers; the physical probe below provides separate board
execution evidence.

The repository-level tool
`examples/BOSCAMEFPGA/vit-base-patch16-224/tools/build_private_ddr_probe.py`
builds a standalone image without submitting it. Its allocator preflight runs
before AME, followed by three native private A/B/P calls with different old
payload poison, cached old CPU reads, CPU/RVV initialization and final reads,
exact i32 checks, frame-gap checks and external output guards. It freezes the
compiler tools and records source/runtime/image hashes, stage commands and ELF
audit beside the image. `--allocator-lock e6-irq` selects the production E6
runtime branch; without marker flags it uses the uninstrumented runtime and
kernel. `--kernel-markers` adds calls outside ACC lifetimes;
`--allocator-markers` instruments a copied diagnostic runtime in the output
directory. Those markers localize a hardware stall and are not performance
measurements or changes to the production allocator.

Both the E6 diagnostic and production probe without allocator/kernel marker
instrumentation completed three poisoned native A/B/P invocations on FPGA4:
all 96 i32 outputs, frame contents/gaps, external guards and four acquire/release
pairs matched. Its observed hart was 0 and incoming MIE was 0; MIE1 behavior has
host-model and instruction evidence.

The uninstrumented production probe task
`afdb4c08032e477fb4c50f58248461fa` used image SHA256
`85b2cc71b40a01606357ffcdd2fc3193899de58d82c4b109d332a09fe1d6f8c2`.
Firmware checked all 96 external outputs and 96 private-P words exactly, plus
frame gaps, guards and allocator/IRQ status. Cycles were 11471, 11348 and
11327; these are probe costs, not a production-kernel speedup. No readmem
output ledger was declared by this probe. Its compiler was the earlier frozen
legal-DDR build; later automatic-profile DDR controls retain that provenance,
with a separately documented final-compiler offline object-identity bridge.
The runtime C/header and compiled runtime object match the production branch.

Normal-SPM old-C probe task `ae078441f80346dd8befc6090adac30a` also passed
16384 downloaded i32 values, input and guard checks. It observed no cached
stale values in any of four calls. Its numerical result is PASS, while
physical stale-cache exposure remains INCOMPLETE; it neither reproduces a
failure without the policy nor establishes that the policy is unnecessary.

These board observations establish the tested target configuration only.
Incoming MIE was zero on board; MIE-one restoration has host-model and native
instruction evidence. The scheduler reported its existing `nh-only` profile;
this does not extend the service's guarantee to arbitrary E6/DMA continuous
reuse. No bitstream/runtime version or measured cycle frequency was supplied.
The final integration acceptance passed 28 unique production/diagnostic tasks
and 40 complete kernel calls for the ten affected default-V2 matmuls, with
8474432 exact firmware i32 comparisons and zero output/input/guard errors.
The integration workspace records task/image/object identities and all thirty
diagnostic rounds in
`plan/validation-evidence/vit-accumulator-repair-final-20261028.json`. Those
matmuls select ACC; the native private-DDR probe and automatic-policy DDR
controls above provide separate evidence of actual private-DDR execution.
No cycle improvement follows from static byte counts or simulator execution.
