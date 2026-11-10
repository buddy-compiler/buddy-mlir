# SpacemiT K3 A100: instruction costs

Measured October 2026 on one A100 core (AI process mode) with
`scripts/spacemit_k3_insn_bench.py`: loops of 16 instructions, independent
destinations unless noted; ns per instruction.

## RVV, e32, LMUL 4 (one `vector<128xf32>` at VLEN 1024)

| Instruction | ns |
| --- | --- |
| `vfmul.vv`, `vfmacc.vv`, `vfmacc.vf` | 4.45 |
| `vfmv.v.f` (broadcast a scalar) | 4.49 |
| `vfredusum.vs`, `vfredmax.vs` | 27.9 |
| `vfredusum.vs`, dependent chain | 56.8 |
| `vrgather.vi` | 17.8 |
| `vslideup.vi` | 17.9 |
| `vfmv.f.s` (element 0 to scalar) | 0.56 |
| `vl4re32.v` from L1 | 10.0 |
| `vs4r.v` (to L1, e.g. a spill) | 11.7-15.8 (varied between runs) |

At e32 mf2 (16 lanes) `vslideup.vi` takes 2.2 ns.

Earlier measurements: `vfredusum` m1 8 ns; `vfwmacc` m1 4.45 ns (`vfwmacc`
at mf2 raised SIGILL in a microbenchmark); `vfmacc` f16 1.11 ns; strided
`vlse32` 307 ns. Vector loads cost ~10 ns each whatever their size up to
LMUL 8: fewer, larger loads are better.

## Consequences

- A reduction per element of a loop (e.g. a dot product per key in decode
  attention) costs ~28 ns each when the reductions are independent and
  ~57 ns when each waits for the previous one: keep them independent.
- Broadcasting a vector element with `vrgather` costs 4 fmas; an
  `extract` to a scalar (0.56 ns) followed by a `.vf` instruction is far
  cheaper. LLVM folds extract + broadcast into `vrgather`: route the value
  through memory or a scalar (see the `riscv-rvv-optimization` skill).
- A spilled LMUL-4 register group costs ~12-16 + 10 ns per store / reload pair.

## IME

- `k3_ime_hp_step` (one 8-column block x 32 rows x one 32-wide K group: three
  loads, unpack, 8 `vmadot.hp`, 4 widening fmas): ~59 ns per group on one
  core alone; ~62.8 ns per group in the 8-core model prefill with the
  activations in TCM; 101.7 ns with everything in DDR; 56.9 ns with
  everything in TCM.
- `smt.vfwmadot` (fp16 8 x 8 times 8 x 8 into f32, 512 multiply-adds;
  `ime.intr.vfmadot`): 0.557 ns per instruction with 12 independent
  accumulators on one core (~920 multiply-adds per ns), 5.57 ns as a
  dependent chain; 1.11 ns each when both cores of a pair run it, 0.557
  ns on cores of different pairs. `smt.vmadot` (int8) also 0.557 ns.
- A tile with one LMUL-4 and one LMUL-8 load per 16 `vfwmadot` (1.5 KiB):
  1.97 ns per `vfwmadot` on one core, 4.3 ns on 4 or 8 cores (the cluster
  load path): convolutions on the IME are bound by their operand loads,
  not by the matrix unit (`models/yolov8`, ~55 ns per K step of 768 bytes
  and 8 `vfwmadot` with 8 cores).

## Memory access patterns and others (October 2026, one A100 core)

| Operation | ns |
| --- | --- |
| `vle64.v` m2 (32 e64, contiguous) | 7.9 |
| `vluxei64.v` m2 (32-element gather) | 306 |
| `vlse64.v` (16 e64, stride 32 bytes) | 151 |
| `vlsseg2e64.v` (8 segments, stride 32 bytes) | 89 |
| `vcompress.vm` e64 m2 | 80 |
| `vrgather.vv` e64 m1 | 38.5 |
| `vsse16.v` (64 e16, stride 64 bytes) | 82 |
| `vsseg8e16.v` (8 x 64 e16) | 289 |
| `vlseg8e32.v` (8 x 32 e32) | 2425 |
| `vsuxei64.v` e32 m2, lanes 8 at a time contiguous | 90 |
| `vsuxei64.v` e32 m2, every lane in another 6.4 KB plane | 676 |
| `vfdiv.vv` e32 m2 | 26.8 |
| `vfmul.vv` e32 m2 | 2.8 |
| `vfrec7.v` e32 m2 | 4.5 |

- Indexed and strided accesses cost ~2-10 ns per element: lay data out so
  that kernels load and store contiguously (`models/yolov8`: column-parity
  split for stride 2, space-to-depth for the first layer).
- An indexed store's cost follows the cache lines its lanes touch; order the
  lanes so that neighbours are contiguous. `llvm.masked.scatter` lowers to
  the ordered `vsoxei`; the RVV intrinsic `llvm.riscv.vsuxei.mask` is
  unordered.
- `vfdiv` is ~10 multiplies: `vfrec7` and Newton steps are cheaper.
- `arith.maximumf` / `minimumf` lower to NaN-propagating sequences
  (`vmfeq` + `vmerge`); `maxnumf` / `minnumf` to one `vfmax` / `vfmin`.
