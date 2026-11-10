# BOSCAME kernel contract, planning and emission

Frontend emits standard Linalg/MemRef/SCF with `contract.*`. Buddy then:

1. `--kernel-contract-verify` — check the neutral contract
2. `--bosc-ame-plan` / `--bosc-ame-plan-verify` — place storage and record `plan.*`
3. `--bosc-ame-emit` — expand a verified plan into BOSCAME / vector ops
4. `--lower-bosc-ame` → LLVM IR

The contract does **not** require Triton or BOSCAME ops. Other frontends can
emit the same IR. Target placement must not leak into the frontend.

Full attribute ABI: [KernelContract.md](KernelContract.md).

## Running the stages

```bash
buddy-opt kernel.ttir \
  --triton-to-linalg-experimental='contract=true caller-contract=args_aligned64_nonalias launch-grid=<sym>:Gx,Gy,Gz' \
  -o kernel.contract.mlir
buddy-opt kernel.contract.mlir --kernel-contract-verify \
  --bosc-ame-plan='plan-report=true' --bosc-ame-plan-verify \
  -o kernel.planned.mlir
buddy-opt kernel.planned.mlir --bosc-ame-emit -o kernel.boscame.mlir
buddy-opt kernel.boscame.mlir --lower-bosc-ame -o kernel.llvm.mlir
buddy-translate kernel.llvm.mlir --buddy-to-llvmir -o kernel.ll
```

- Use the real entry symbol and launch grid. Grid specializes full/tail
  programs; it does **not** authorize masked-row access.
- `contract=false` (default) keeps ordinary target-independent lowering.
- Handwritten buffer Linalg: `--kernel-contract-annotate`, then verify/plan.
- `--linalg-to-boscame`share the same contract → plan → emit path
(V1 recipe subset only).
- No global `store-policy`. Remove legacy `bosc_ame.store_policy`; planner
  chooses per-operand routes (e.g. SPM A/C + direct DDR B).

## Layers (do not conflate)

| Object | Meaning |
| --- | --- |
| Standard op | Computation and its real memory reads/writes |
| Contract task / scope | Dependencies, completion, modeled control flow |
| Content version | Bytes produced by a writer (incl. partial-write inheritance) |
| Logical root / view | Alloc or arg identity, offset, sizes, strides |
| Physical window | Bank, byte offset, capacity for a private root |
| Access route | Concrete DDR / SPM / host path for an operand |

Planner binds operands, drops only **proven** redundant storage events, and
rebuilds tasks/versions. `plan.task_map` must cover every source task;
emission must not revive eliminated tasks as executable events.

## Placement and routes (summary)

Private accumulation across K stays private. Capacity may push A/B to another
**already legal** direct route; it cannot drop required snapshots or move
external writes without proof.

Boundary copies cover exact effective views on a shared contiguous axis.
Engine choice is automatic and deterministic:

1. Pure DMA if wholly legal (`engine="dma"`, 64 B DMA alignment per
   `SpmDma.h`)
2. Else matrix ACC32 transport (`bosc_ame.matrix.copy`, `engine="matrix"`)
3. Else segmented DMA + RVV (`engine="segmented"`)

Each candidate proves its own endpoints and coverage. Failed matrix does not
weaken DMA checks. There is no uncalibrated latency ranking.

- **Matrix copy**: raw e32 ACC32 DDR↔SPM or SPM↔SPM; no arithmetic; closed
  type set; exact row access (no packing/transpose). Tile bound: ≤64 rows,
  ≤256 B/row, ≤16384 B. See also [BOSCAMEPrivateDDR.md](BOSCAMEPrivateDDR.md).
- **Deferred zero fill** (proved private i32): may become
  `bosc_ame.tile.zero_acc` on first use — [BOSCAMEZeroAccumulator.md](BOSCAMEZeroAccumulator.md).
- Descriptor offsets are in **elements**; transfer lengths / pitches /
  windows are in **bytes**. Do not apply offsets twice.

ABI: `contract.abi` only on unused-entry functions. Emission records
`args_aligned64` or `args_aligned64_nonalias`. Alignment alone does not prove
concurrent DDR disjointness.

## Current emission subset

- AME: static signed i8×i8→i32 matmul; C load/store each time; ACC not live
  across outer K or DMA runtime calls.
- Other matmuls: explicit scalar `plan.compute` recipe (ordered M/N/K, no
  reassociation).
- Pointwise: equal-shaped contiguous rank-1, pure scalar yield; mixed widths
  share one lane count.
- Ordered f32 sum / maxnum reductions: fixed chunks; LayerNorm still uses
  `rtol=atol=2e-5`.
- Host rank-0 buffers: static, private, outside repeated scopes → stack.
- SPM: no calls in functions with private/fixed SPM windows; no module-local
  symbol uses of such functions. DDR-only calls remain OK.
- Physical overlap of different roots is checked; without liveness reuse
  proof, only mutually exclusive arms may share windows.

Unsupported shapes/routes diagnose the failed constraint; emission does not
invent a new route after verify fails.

### Explicit math approximation

Requires per-function `contract.math` authorization (domain, IEEE specials,
abs/rel error). Planner records `plan.vector.math`; verifier rejects missing
or forged entries. No target/SFU type in the neutral contract.

Example (ViT):

```mlir
contract.math = {erf = {domain = "all-f32", special_values = "ieee",
  max_abs_error = 2.0e-6 : f64, max_rel_error = 0.0 : f64}}
```

FPGA profile may emit scalar polynomial approximations under that budget;
output tolerances stay `2e-5` unless separately changed.

## Evidence pointers

Native fixtures (planner/emitter smoke):

| Fixture | SPM windows | Workspace B | Surviving tasks |
| --- | ---: | ---: | ---: |
| 196×768×768 i8 matmul | 2 | 20480 | 10 |
| 128×1024×128 i8 matmul | 0 | 0 | 1 |
| 197×768 LayerNorm | 6 | 24576 | 21 |

Production `make check` / `simcheck` stack bounds and traffic hashes:
[BOSCAMEKernelContractValidation.md](BOSCAMEKernelContractValidation.md).

Zero-acc / W8A8: [validation/zero-acc-w8a8/README.md](validation/zero-acc-w8a8/README.md).
Nonlinear math: [validation/nonlinear-math/README.md](validation/nonlinear-math/README.md).

Simulator time is not hardware performance. Board results apply only to the
compiler outputs they recorded.
