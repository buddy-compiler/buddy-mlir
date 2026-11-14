# BOSCAME dialect examples

Small host-side programs under this directory exercise:

- `generic-*` and `linalg-to-boscame-matmul.mlir` — keep mathematical inputs
  through standard Linalg→LLVM reference lowering; lit checks printed values
  (broadcasts and payloads outside the V1 FPGA planner contract).
- Handwritten `bosc-*` programs — standalone `bosc_ame` instruction lowering.

These are **not** the NR board path. Board examples and firmware live in
[`../BOSCAMEFPGA/`](../BOSCAMEFPGA/README.md). Shared runtime sources live in
[`../../runtime/boscame/`](../../runtime/boscame/README.md).

## Docs map

| Doc | Topic |
| --- | --- |
| [`docs/KernelContract.md`](../../docs/KernelContract.md) | Neutral buffer contract |
| [`docs/BOSCAMEKernelContractPipeline.md`](../../docs/BOSCAMEKernelContractPipeline.md) | Contract → plan → emit |
| [`docs/BOSCAMEPrivateDDR.md`](../../docs/BOSCAMEPrivateDDR.md) | Private DDR frame |
| [`docs/BOSCAMEZeroAccumulator.md`](../../docs/BOSCAMEZeroAccumulator.md) | Deferred zero-acc proof |

Op/type definitions: `midend/include/Dialect/BOSCAME/BOSCAME.td`.
Lit coverage: `tests/Conversion/BOSCAME`, `tests/KernelContract`.
