# SpacemiT IME Dialect Usage Guide

This document provides a comprehensive guide for using the IME (Intelligent Matrix Extension) dialect in buddy-mlir.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Operations Reference](#operations-reference)
4. [Matrix Dimensions](#matrix-dimensions)
5. [Example Workflow](#example-workflow)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Introduction

The IME dialect provides MLIR operations that map to SpacemiT's Intelligent Matrix Extension for RISC-V. It enables high-performance matrix computations commonly used in AI/ML workloads.

### Key Features

- **5 Core Operations**: vmadot, vmadotu, vmadotsu, vmadotus, vfmadot
- **Mixed Precision**: int8→int32 and fp16→fp32 accumulation
- **Signedness Variants**: Handle signed/unsigned combinations correctly
- **MLIR Integration**: Full integration with buddy-mlir optimization pipeline

## Quick Start

### 1. Build buddy-mlir with IME Support

```bash
cd buddy-mlir
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
ninja
```

### 2. Write an IME MLIR Program

```mlir
// simple.mlir
func.func @matmul(%a: memref<4x8xi8>, %b: memref<8x4xi8>, %c: memref<4x4xi32>) {
  ime.vmadot %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  return
}
```

### 3. Compile to RISC-V

```bash
# Lower IME and other dialects
buddy-opt simple.mlir \
    --lower-ime \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o lowered.mlir

# Translate to LLVM IR
buddy-translate lowered.mlir --buddy-to-llvmir -o output.ll

# Generate RISC-V assembly
llc output.ll \
    -mtriple=riscv64-unknown-linux-gnu \
    -mattr=+m,+v,+xtheadime \
    -filetype=asm \
    -o output.s
```

## Operations Reference

### ime.vmadot

Signed int8 × Signed int8 → int32 accumulator

```mlir
ime.vmadot %accumulator, %matrix_a, %matrix_b : 
    memref<MxNxi32>, memref<MxKxi8>, memref<KxNxi8>
```

**Semantics**: `accumulator += matrix_a @ matrix_b`

### ime.vmadotu

Unsigned int8 × Unsigned int8 → int32 accumulator

```mlir
ime.vmadotu %accumulator, %matrix_a, %matrix_b : 
    memref<MxNxi32>, memref<MxKxi8>, memref<KxNxi8>
```

### ime.vmadotsu

Signed int8 × Unsigned int8 → int32 accumulator

```mlir
ime.vmadotsu %accumulator, %matrix_a, %matrix_b : 
    memref<MxNxi32>, memref<MxKxi8>, memref<KxNxi8>
```

### ime.vmadotus

Unsigned int8 × Signed int8 → int32 accumulator

```mlir
ime.vmadotus %accumulator, %matrix_a, %matrix_b : 
    memref<MxNxi32>, memref<MxKxi8>, memref<KxNxi8>
```

### ime.vfmadot

FP16 × FP16 → FP32 accumulator

```mlir
ime.vfmadot %accumulator, %matrix_a, %matrix_b : 
    memref<MxNxf32>, memref<MxKxf16>, memref<KxNxf16>
```

## Matrix Dimensions

IME operates on fixed tile sizes determined by VLEN (Vector Register Length):

| VLEN | Data Type | M | K | N | Description |
|------|-----------|---|---|---|-------------|
| 256 | int8 | 4 | 8 | 4 | 4×8 × 8×4 → 4×4 |
| 256 | fp16 | 4 | 4 | 4 | 4×4 × 4×4 → 4×4 |
| 128 | int8 | 4 | 4 | 4 | 4×4 × 4×4 → 4×4 |
| 128 | fp16 | 4 | 2 | 4 | 4×2 × 2×4 → 4×4 |

For SpacemiT X100 (VLEN=256):
- **Integer operations**: Use 4×8 and 8×4 input matrices
- **FP16 operations**: Use 4×4 input matrices

## Example Workflow

### Step 1: Define Input Data

```mlir
memref.global "private" @weights : memref<4x8xi8> = dense<[
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8]
]>

memref.global "private" @activations : memref<8x4xi8> = dense<1>
```

### Step 2: Perform Matrix Multiplication

```mlir
func.func @inference() -> memref<4x4xi32> {
  %weights = memref.get_global @weights : memref<4x8xi8>
  %activations = memref.get_global @activations : memref<8x4xi8>
  
  %output = memref.alloc() : memref<4x4xi32>
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%output : memref<4x4xi32>)
  
  ime.vmadot %output, %weights, %activations : 
      memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  
  return %output : memref<4x4xi32>
}
```

### Step 3: Compile and Run

```bash
# Full compilation pipeline
buddy-opt model.mlir \
    --lower-ime \
    --convert-linalg-to-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-math-to-llvm \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts | \
buddy-translate --buddy-to-llvmir | \
llc -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+v,+xtheadime -filetype=asm -o model.s
```

## Advanced Usage

### Tiled Matrix Multiplication

For large matrices, tile the computation:

```mlir
func.func @tiled_matmul(%A: memref<16x32xi8>, 
                        %B: memref<32x16xi8>, 
                        %C: memref<16x16xi32>) {
  // Tile sizes: M_TILE=4, K_TILE=8, N_TILE=4
  affine.for %i = 0 to 16 step 4 {
    affine.for %j = 0 to 16 step 4 {
      affine.for %k = 0 to 32 step 8 {
        // Extract tiles
        %a_tile = memref.subview %A[%i, %k][4, 8][1, 1] 
            : memref<16x32xi8> to memref<4x8xi8, strided<[32, 1], offset: ?>>
        %b_tile = memref.subview %B[%k, %j][8, 4][1, 1]
            : memref<32x16xi8> to memref<8x4xi8, strided<[16, 1], offset: ?>>
        %c_tile = memref.subview %C[%i, %j][4, 4][1, 1]
            : memref<16x16xi32> to memref<4x4xi32, strided<[16, 1], offset: ?>>
        
        // IME computation on tile
        ime.vmadot %c_tile, %a_tile, %b_tile : 
            memref<4x4xi32, strided<[16, 1], offset: ?>>,
            memref<4x8xi8, strided<[32, 1], offset: ?>>,
            memref<8x4xi8, strided<[16, 1], offset: ?>>
      }
    }
  }
  return
}
```

### Integration with Linalg

Future work includes automatic conversion from `linalg.matmul` to IME operations.

## Troubleshooting

### Common Issues

1. **"ime.vmadot' op not registered"**
   - Ensure buddy-opt was built with IME dialect support
   - Check that LowerIMEPass is linked in `midend/lib/CMakeLists.txt`

2. **"unknown attribute 'xtheadime'"**
   - LLVM needs to be built with IME extension support
   - Use a patched LLVM from SpacemiT or the buddy-mlir submodule

3. **Incorrect Matrix Dimensions**
   - Verify tile sizes match your VLEN configuration
   - For VLEN=256: int8 uses 4×8 × 8×4, fp16 uses 4×4 × 4×4

### Debug Commands

```bash
# Check if IME dialect is registered
buddy-opt --show-dialects 2>&1 | grep ime

# Verify pass is available
buddy-opt --help | grep lower-ime

# Debug lowering
buddy-opt input.mlir --lower-ime --debug 2>&1 | less
```

## File Locations

| Component | Path |
|-----------|------|
| Dialect Definition | `midend/include/Dialect/IME/IME.td` |
| C++ Headers | `midend/include/Dialect/IME/IMEDialect.h`, `IMEOps.h` |
| Implementation | `midend/lib/Dialect/IME/IR/IMEDialect.cpp` |
| Lowering Pass | `midend/lib/Conversion/LowerIMEPass.cpp` |
| Transforms | `midend/lib/Dialect/IME/Transforms/` |
| Examples | `examples/IMEDialect/` |

## Related Documentation

- [IME Extension Specification](../riscv-ime-extension-spec/README.md)
- [IME Complete Instruction Flow](IME_Complete_Instruction_Flow.md)
- [IME vs RVV Instructions](IME_vs_RVV_Instructions.md)
- [IME Instructions Reference](IME_Instructions_Reference.md)
