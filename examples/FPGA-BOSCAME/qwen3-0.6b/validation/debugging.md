# FPGA execution issues resolved during validation

## Triton tail case: unintended RVV in C descriptor initialization

The AME suite passed the eleven regular matrix cases, then stopped after
`matmul_3x19x70 BEGIN`. The masked Triton kernel itself was unchanged and its
host oracle passed all 57 outputs.

An isolated diagnostic image printed `launch entry` and `init complete`, but
never reached `before ame_fence`. Its linked assembly placed the following
instructions between those messages, while initializing the three `MemRef2`
descriptors in `launch.c`:

```asm
0x80012120: jal nr_puts                 # init complete
...
0x8001212c: vsetivli zero, 4, e64, m1, ta, ma
0x80012130: vmv.s.x v8, a0
...
0x8001214c: vsext.vf8 v11, v8
0x80012156: vse64.v v11, (a0)
...
0x8001219c: jal nr_puts                 # before ame_fence; never reached
```

The diagnostic image SHA-256 was
`446c02408072afe08fdb5c70007757bffb9a9db147ba51d02a59c06446e11c6d`.
These addresses identify that image only. LLVM selected RVV instructions for
fixed-size C aggregate initialization because the C compiler was given `+v`;
disabling loop and SLP vectorization did not prevent this selection. The
generated forms included instructions outside the verified NR subset. The
program stopped before AME synchronization, the adapter call, or the operator.

The fix compiles all C launchers, adapters, and runtime code with
`rv64gc_zicbom`. Buddy-generated operator assembly retains RVV and passes the
NR instruction audit. The final linked ELF is also audited, so vector code
introduced outside `kernel.s` cannot escape inspection.

With only that C target correction, the original masked tail kernel passed
all 57 outputs on FPGA in run `run-6c330555d4cb472b`. No BM/BK change or added
kernel tracing was needed for the successful production run.

The diagnostic tracing saved and restored caller-saved integer registers and
introduced console traffic and fences; it was used to locate the stop, not as
the correctness result. A separate functional interpretation of the original
generated assembly also completed both grid programs, checked all 57 outputs,
and preserved the stack and return address. That interpretation checked scalar
control flow and ideal AME semantics, not hardware timing or coherence.

## FP32 linear: whole-register move depends on VL on this board

The first full RVV suite passed all attention cases, but the transpose-B linear
path produced incorrect sums. The same vector-lowered LLVM passed its host
oracle. Generated assembly hoisted the zero vector outside the dot-product
loop and initialized each accumulator with:

```asm
vsetivli zero, 1, e8, m1, ta, ma
vmv1r.v v9, v8
```

The earlier capability probe checked `vmv1r.v` only at e32/m1/VL16. A follow-up
loaded distinct source and destination values, ran the exact e8/VL1 sequence,
then compared all 16 e32 lanes. All 16 were wrong. This board's operation cannot
be treated as the standard VL-independent whole-register copy. The evidence is
run `run-baf62e6e7b5242a1`, image SHA-256
`f07bb95c4cd904a4e22fd1d86e2fd60d01b1f13cfb890f0a78f4ef6e88760fc1`.

NR FP32 linear code generation now uses `-disable-machine-licm -disable-machine-cse`.
LLVM emits an explicit e32/VL16 zero initialization inside each dot, followed by
vector FMA. All whole-register move forms are rejected by the assembly and
final ELF checks; no assembly rewrite substitutes another implementation.

The corrected production `matmul_1x1024x1024_f32` passed all 1024 outputs with
zero error on FPGA in run `run-c45fadb38667484b`, image SHA-256
`a666bffbfea7771cdcefce566cb4b0b5a6d27547b8f3988ce3df3c49d408d6d8`.
Its kernel consumed `0x2e5758` cycles. Final suite evidence is recorded separately.

## Attention PV decode: keep its independently validated machine optimizations

Applying the linear workaround globally changed the scalar scheduling and
register allocation around attention's RVV loop. In suite image
`9eb6f0850e67040f6e89daddb4f171ef0377f2d3f09c917edfddb91067f048fc`,
`attention_pv_16x1x128x17` failed 1876 of 2048 outputs, with maximum absolute
error 0.375. The default machine-optimization version had passed on FPGA.

The [assembly comparison](attention-codegen/README.md) preserves both versions.
Both use e32/m1/VL16, load the accumulator, execute 17 vector FMA iterations,
and store the result. Neither introduces a tail VL or a whole-register move.
The differences are in scalar loop-constant hoisting, addresses, GPR allocation,
and stack saves. The hardware cause is not established; this is currently a
code-scheduling-sensitive observation, not proof of a reduction-tail defect.

The shared configuration therefore separates `RVV_FLAGS` from
`RVV_LINEAR_FLAGS`. Attention retains the default machine optimizations;
only FP32 linear matmul applies the two disabling flags to avoid its proven
whole-register-copy failure. Both paths still execute Buddy vector lowering,
full-output oracles, and final ELF ISA/fence audits. CPU execution of identical
LLVM IR cannot substitute for rerunning the resulting machine code on FPGA.
