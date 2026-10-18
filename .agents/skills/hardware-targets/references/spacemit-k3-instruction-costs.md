# SpacemiT K3 A100: instruction costs

Measured October 2026 on one A100 core (AI process mode) with
`scripts/spacemit-k3-insn-bench.py`: loops of 16 instructions, independent
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
