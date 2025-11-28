# IME Dialect Examples

This directory contains examples demonstrating the usage of the IME (Intelligent Matrix Extension) dialect in buddy-mlir.

## Overview

The IME dialect provides operations that map to SpacemiT's Intelligent Matrix Extension (IME) for RISC-V. IME accelerates matrix operations commonly used in AI/ML workloads through specialized vector instructions.

## Operations

The IME dialect supports the following operations:

| Operation | Description | Input Types | Accumulator Type |
|-----------|-------------|-------------|------------------|
| `ime.vmadot` | Signed × Signed matrix multiply-accumulate | i8 × i8 | i32 |
| `ime.vmadotu` | Unsigned × Unsigned matrix multiply-accumulate | i8 × i8 | i32 |
| `ime.vmadotsu` | Signed × Unsigned matrix multiply-accumulate | i8 × i8 | i32 |
| `ime.vmadotus` | Unsigned × Signed matrix multiply-accumulate | i8 × i8 | i32 |
| `ime.vfmadot` | FP16 × FP16 matrix multiply-accumulate | f16 × f16 | f32 |

## Examples

### vmadot-basic.mlir

Basic integer matrix multiplication using `ime.vmadot`:

```mlir
ime.vmadot %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
```

This performs: `C = C + A × B` where:
- `A` is a 4×8 signed int8 matrix
- `B` is an 8×4 signed int8 matrix
- `C` is a 4×4 signed int32 accumulator

### vfmadot-basic.mlir

Floating-point matrix multiplication using `ime.vfmadot`:

```mlir
ime.vfmadot %c, %a, %b : memref<4x4xf32>, memref<4x4xf16>, memref<4x4xf16>
```

This performs: `C = C + A × B` where:
- `A` is a 4×4 fp16 matrix
- `B` is a 4×4 fp16 matrix
- `C` is a 4×4 fp32 accumulator

### vmadot-variants.mlir

Demonstrates all four integer vmadot variants with different signedness combinations.

## Building the Examples

### Prerequisites

1. Build buddy-mlir with IME dialect support
2. Ensure `buddy-opt`, `buddy-translate`, and `llc` are available

### Quick Start

```bash
# From the examples/IMEDialect directory
cd examples/IMEDialect

# Compile all examples
make all

# Or compile individual examples
make vmadot-basic
make vfmadot-basic
make vmadot-variants

# Check IME lowering without generating assembly
make check-vmadot-basic

# Clean generated files
make clean
```

### Manual Compilation Steps

1. **Lower IME dialect to LLVM dialect:**
```bash
buddy-opt input.mlir \
    --lower-ime \
    --convert-linalg-to-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-math-to-llvm \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o lowered.mlir
```

2. **Translate to LLVM IR:**
```bash
buddy-translate lowered.mlir --buddy-to-llvmir -o output.ll
```

3. **Generate RISC-V assembly with IME instructions:**
```bash
llc output.ll \
    -mtriple=riscv64-unknown-linux-gnu \
    -mattr=+m,+v,+xtheadime \
    -filetype=asm \
    -o output.s
```

## Matrix Dimensions

IME operations work on tile-based matrix multiplication. The tile size depends on VLEN (Vector Length):

| VLEN | SEW=8 (int8) | SEW=16 (fp16) |
|------|--------------|---------------|
| 128 | 4×4 tiles | 4×2 tiles |
| 256 | 4×8 tiles | 4×4 tiles |
| 512 | 4×16 tiles | 4×8 tiles |

For a VLEN=256 configuration (common in SpacemiT X100):
- Integer operations: A is 4×8, B is 8×4, C is 4×4
- FP16 operations: A is 4×4, B is 4×4, C is 4×4

## Related Documentation

- [IME Extension Specification](../../riscv-ime-extension-spec/README.md)
- [IME Complete Instruction Flow](../../docs/IME_Complete_Instruction_Flow.md)
- [IME vs RVV Instructions](../../docs/IME_vs_RVV_Instructions.md)
- [IME Instructions Reference](../../docs/IME_Instructions_Reference.md)

## Notes

- The IME dialect is designed to work with SpacemiT's X100 series processors
- LLVM must be built with IME extension support for final code generation
- The `--lower-ime` pass converts IME operations to LLVM intrinsics or inline assembly
