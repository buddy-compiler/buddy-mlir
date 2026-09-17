# AME v0.5 direct-GPR contract probe

`ame_gpr_probe.c` is a standalone instruction probe for the second optimization.
It does not replace a Triton kernel or provide model inference. Link it with
the existing `common/nr` runtime, using `NR_CFLAGS` and the existing NR linker.
Run the final ELF through `tools/check_nr_elf.py` before using `fpga_run.sh`.
Do not combine this launch object with a second `launch` implementation.

The probe exercises:

| Instruction | Actual encoded GPRs | Matrix register |
| --- | --- | --- |
| `msettilem` | `rd = rs1 = x18` | none |
| `msettilen` | `rd = rs1 = x23` | none |
| `msettilek` | `rd = rs1 = x31` | none |
| `mlae8.m` | base `x18`, byte stride `x19` | `tr0` |
| `mlbe8.m` | base `x20`, byte stride `x21` | `tr4` |
| `mlce32.m` | base `x22`, byte stride `x23` | `acc0` |
| `msce32.m` | base `x30`, byte stride `x31` | `acc0` |

Every source GPR in the load/store probe uses the high half of the 32-register
file, exercising the full five-bit fields. The low MLS field is a matrix
register, not a GPR clobber. The actual inline-asm operands use explicit register
variables and input/output constraints; no undocumented temporary or SP write
is introduced. Tile return values must be exactly 2, 3 and 4, respectively.

The signed INT8 example is the v0.5 software manual's `M=2, N=3, K=4`, with
nonzero signed INT32 accumulator input. Round zero must produce
`[[16, -18, 26], [-2, 35, -17]]`. B uses ordinary `[N,K]` layout. A/B have byte
strides 16/32; C/output have byte strides 32/64. All row padding and front/back
guards are checked. Input arrays and initial accumulators must remain unchanged.
The same buffers are reinitialized with different A/B/C values for four rounds,
and every result is compared with an independent integer oracle.

Both fences around every AME instruction remain present. `msettype` retains
the safe instruction/clobber contract from the shared runtime. The probe calls
the existing public `ame_fence()` before and after every round, so it does not
reduce the completion/resynchronization policy. It does not attempt to establish
a general cache-maintenance contract; it only tests the current NR environment.

Acceptance requires four round lines with `errors=00000000`, tile returns
2/3/4, final `verify AME direct GPR probe: PASS`, and normal successful runtime
return. A successful instruction probe must still be followed by real
Triton/Buddy matmul A/B tests with the `fixed` and `direct` encoder modes.
This file records the intended test; hardware results belong in validation
records after the root agent executes the probe.
