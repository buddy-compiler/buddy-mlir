# K64 reissue audit

Date: 2026-09-22 UTC

This note checks whether `references/ame-k64-reissue-current-lowering.patch`
is missing from the code path used by the current 28-layer Qwen3 image.  It is
a static audit; it does not claim that reissuing K eliminates every AME tile
state or timing problem.

## What the patch changes

The patch targets
`midend/lib/Conversion/LowerQwenW8A8ToBOSCAME/LowerQwenW8A8ToBOSCAME.cpp`.
It removes the operation-wide `MSettilekOp(..., 64)` and emits
`MSettilekOp(..., 64)` after each fresh accumulator group has been loaded and
immediately before the reduction loop's first A tile.  The exact insertion
points are one `emitTokenTile` insertion in the scalar fallback, plus one
insertion in each optimized `emit1A4B`, `emit2A4B`, and `emitDecodePair` helper.
`W8A8LinearPairLowering` itself only drops its shared operation-wide K write;
its child `createLinear` calls use those per-group sites.  The pair class does
not add a fifth independent K schedule.

The patch file is
[`ame-k64-reissue-current-lowering.patch`](/home/chh/gitprojects/buddy-mlir-for-fpga/references/ame-k64-reissue-current-lowering.patch).
The current source still has the old operation-wide K writes at lines 934--938
and 1359--1361 and no K write between the accumulator loads and K loop in the
1A4B, 2A4B, and decode-pair helpers (for example, lines 1677--1687,
1781--1797, and 1933--1948):

```text
mlce32/load accumulator(s)
configureMmaType(i8)
K loop -> load A
```

Thus the patch is genuinely absent from this direct semantic lowering source.

## Path used by the current 28-layer image

The production build under
`examples/FPGA-BOSCAME/qwen3-0.6b/model/build/quant-opt/shared/` uses the
Triton-riscv kernel archive, then lowers the model graph to calls into that
archive.  The generated graph files
`nr-prefill/forward_prefill.ll`, `nr-decode/forward_decode.ll` and their
`forward_*.nr.S` outputs contain calls such as
`qwen_graph_w8a8_mm_matmul_16x2048x1024`; the `forward_*.nr.S` files do not
contain AME words themselves.  The AME words are in the archived kernel
objects and their evidence assembly under
`shared/model-lib/evidence/`.

The archive manifest reports 46 kernel cases.  They divide as follows:

| cases | count | K-reissue relevance |
| --- | ---: | --- |
| all evidence `kernel.nr.S` files | 46 | complete archive case set |
| integer AME matmul (`matmul_*`) | 11 | five M=16 and six M=1 kernels; these issue K |
| non-matmul kernels | 35 | no AME matmul accumulator, so no K reissue is expected |

The 11 matmul cases are
`matmul_16x1024x1024`, `matmul_16x1024x2048`,
`matmul_16x1024x3072`, `matmul_16x2048x1024`,
`matmul_16x3072x1024`, and
`matmul_1x1024x1024`, `matmul_1x1024x2048`,
`matmul_1x1024x3072`, `matmul_1x151936x1024`,
`matmul_1x2048x1024`, `matmul_1x3072x1024`.

## K control-flow evidence in all 11 matmul kernels

Every one of those 11 `kernel.ll` files has the same control-flow shape:

```text
msettype(i32)
mlce32.m                         # load the accumulator seed
msettype(i8)
K-loop:
  k = min(remaining_K, 64)
  msettilek(k)
  mlae8.m                         # first A load follows immediately
  ... mma ...
```

For the model dimensions in the archive (`K` is 1024, 2048, or 3072), the
first K-loop value is exactly 64; all are divisible by 64.  The K instruction
is in the loop body, so it is reissued before every A tile, including the first
one after the accumulator load.  The static `.nr.S` has one K word because the
word is the loop body, not because it executes only once.

Representative exact lines are:

- [`matmul_16x2048x1024/kernel.ll`](/home/chh/gitprojects/buddy-mlir-for-fpga/examples/FPGA-BOSCAME/qwen3-0.6b/model/build/quant-opt/shared/model-lib/evidence/matmul_16x2048x1024/kernel.ll:96): `mlce32`, `msettype(i8)` at 97, `msettilek(smin(...,64))` at 109, and `mlae8` at 125.
- [`matmul_1x1024x3072/kernel.ll`](/home/chh/gitprojects/buddy-mlir-for-fpga/examples/FPGA-BOSCAME/qwen3-0.6b/model/build/quant-opt/shared/model-lib/evidence/matmul_1x1024x3072/kernel.ll:134): the same sequence at lines 134--163.
- [`matmul_16x2048x1024/kernel.nr.S`](/home/chh/gitprojects/buddy-mlir-for-fpga/examples/FPGA-BOSCAME/qwen3-0.6b/model/build/quant-opt/shared/model-lib/evidence/matmul_16x2048x1024/kernel.nr.S:150): the encoded K word is at line 156 and the following A word at line 161.
- [`matmul_1x1024x3072/kernel.nr.S`](/home/chh/gitprojects/buddy-mlir-for-fpga/examples/FPGA-BOSCAME/qwen3-0.6b/model/build/quant-opt/shared/model-lib/evidence/matmul_1x1024x3072/kernel.nr.S:222): the encoded K word is at line 228 and the following A word at line 233.

The other nine matmul kernels have the same relative ordering; their
`kernel.ll` files are in the same `evidence/matmul_*/` directory.  Therefore,
for the actual quant-opt archive, there is no static omission where a fresh C
accumulator load is followed by an A load without a K write.  The K write is
after the i8 type switch in this path; the direct Qwen patch places it before
that switch.  Both are before the first A load.  If hardware requires the
stronger ordering "K before msettype(i8)", that would be a separate experiment
and is not implied by this audit.

The existing reference screenshot `references/fence/ame_fence.png` likewise
shows `mlce32` before the depth loop, then `msettype(AME_MTYPE_I8_MMA)` followed
by `msettilek(tile_k)` before `mlae8`. This is another example of type-before-K,
not an authoritative hardware guarantee or a board result for the present
image. The stronger order requirement still needs confirmation.

## Existing LowerLinalg K path

The Triton-riscv kernels used by the current image are lowered through the
Linalg-to-BOSCAME path, where the NR-specific branch reissues K in the K-loop
at
[`LowerLinalgToBOSCAME.cpp:1766`](/home/chh/gitprojects/buddy-mlir-for-fpga/midend/lib/Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.cpp:1766),
immediately before the first A load at line 1768.  The generic branch also has
a `configureTileK` at line 1618, but that is not the NR branch used for these
archive kernels and is not counted as the evidence here.  The LowerLinalg
source hash is:

```text
LowerLinalgToBOSCAME.cpp  6830de160b1c9185cba4041e23dba40b6591d22ef4a4c16c174dc5a3e514bced
```

## Encoders and graph entry assembly

[`ame_to_word.py`](/home/chh/gitprojects/buddy-mlir-for-fpga/examples/FPGA-BOSCAME/tools/ame_to_word.py)
encodes an existing `msettilek` and does not schedule, remove, or move it.
[`restrict_fpga_assembly.py`](/home/chh/gitprojects/buddy-mlir-for-fpga/examples/FPGA-BOSCAME/tools/restrict_fpga_assembly.py)
validates the raw word and inserts surrounding `fence rw,rw`; it likewise does
not change K scheduling.  The graph entry assembly only calls the archive
symbols, so applying the direct `LowerQwenW8A8ToBOSCAME` patch alone would not
change the current quant-opt image.

## Artifact identities

SHA256 values for this audit:

```text
LowerQwenW8A8ToBOSCAME.cpp  2435a6df62257ea74043c04e0cf240be6e6beedb9d263e087a2f99466e7393e2
ame-k64-reissue-current-lowering.patch
                             33d302081f50bcf227ac25bac660d181e64b06d9ad0fc9f4e949449bc629388a
shared/model-lib/archive.json
                             d430b9cdbbeac467b9ed59eaedbd04a69061f565ace436e52855d2a5ed26179f
shared/model-lib/libqwen_triton.a
                             9e45eb8832b3bf8c0c578de762c684711f0030bcba78cac74b2bd0ee6e0fbe34
shared/nr-prefill/forward_prefill.ll
                             1f04f5426877d94e12f4a818c2920cb70181cb84c8393ea4cd7fc83694dbea64
shared/nr-decode/forward_decode.ll
                             1de21943c89fb0846168bf713e481312460cd8922296ead6ee86cf9d6b1f5e2a
```

This establishes schedule coverage in the current archive.  It does not
establish that AME internal state is correct for every startup sequence, cache
state, register encoding, or hardware timing condition.

## Graph-final helper and hardware-contract follow-up

The non-kernel AME path in `common/nr/ame_sync.c` is a 1x1 dummy operation:

```text
M=1, N=1, K=1 -> i8 type -> A -> B -> MMA -> i32 type -> C load -> C store
```

Disassembly of the production-control ELF confirms the single A load after
K=1 (`ame_fence` at `0x801703ec`, size `0xbc`). K is issued at `0x8017041e`,
i8 type at `0x80170430`, A at `0x80170442`, and C at `0x80170486`. There is
no second A load after that C load within the helper. The next helper issues
K=1 again; the next matmul issues K=64 after its own C seed and before A.
Do not substitute K=64 into this 1x1 helper with one-byte A/B objects.

The workspace's AME v0.5 documentation describes dimension setters and MxN
accumulator loads, but does not establish that MLCE32 changes K, nor describe
all effects of the legacy `msettype` values used here. ModelZoo's software
`k_after_c` assertion is a software contract, not independent RTL evidence.
The following still need hardware-design confirmation:

1. Whether C -> K64 -> i8 -> A and C -> i8 -> K64 -> A are equivalent.
2. Whether legacy msettype clips, resets, or latches existing K state, and
   whether MLCE32 invalidates that state.
3. Whether the helper's first dummy MMA may legally use an accumulator that
   has not yet been initialized; its result is subsequently overwritten.

No supplied patch has been applied to production code, and no modified K
schedule has been claimed as a tested fix.
