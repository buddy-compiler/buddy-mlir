# XuanTie AME Dialect

This document provides a comprehensive guide for using the AME (Attached Matrix Extension) dialect in buddy-mlir.

## Table of Contents

1. [File Structure](#file-structure)
2. [Introduction](#introduction)
3. [Quick Start](#quick-start)
4. [Reference](#reference)
5. [Note](#note)

## File Structure

The XTAMEDialect example folder contains the following key files:

**MLIR Source Files:**

- **`ame-th-mmacc-w-b.mlir`**: Minimal MLIR files demonstrating each basic AME operation. Use these to verify assembly code generation.
- **`run-ame-th-mmacc-w-b.c`**: Extended test .c file with additional code for printing matrix results on hardware. Use these files for joint assembly to generate an executable for visible hardware verification.

**Build Files:**

- **`makefile`**: Build automation for generating lowered MLIR, LLVM IR, and assembly code.

## Introduction

The AME dialect provides MLIR operations that map to XuanTie's RISC-V matrix extensions.

### Matrix Operations

All AME instructions perform the following core matrix multiply-accumulate operation:

```
C (M×N) += A (M×K) × BT (N×K)
```

Where:

- **M, N, K**: Matrix dimensions (determined by RLEN configuration)

  1. shape of matrix A: M rows(sizeM), K columns(sizeK x 8 / MSEW)

  2. shape of matrix B: N rows(sizeN), K columns(sizeK x 8 / MSEW)

  3. shape of matrix C: M rows(sizeM), N columns(sizeN)

- **A (ms1)**: Left operand matrix (signed int8/unsigned int8 or fp8/fp16/bf16/fp32/fp64)

- **B (ms2)**: Right operand matrix (signed int8/unsigned int8 or fp8/fp16/bf16/fp32/fp64)

- **C (md)**: result matrix (signed int32/unsigned int32 for integer ops, fp16/fp32/fp64/bf16 for float ops)

- **RLEN：**128bits、256bits、512bits（If the size is larger, it will need to be divided into sections.）


## Quick Start

### Prerequisites

Before you start, ensure you have the following prerequisites installed:

1. **xuantie-gcc**

   Cross-compile the combined assembly file .s and the print file .c to generate an executable file.

2. **xuantie-qemu**

   Execute the generated executable file to obtain the results.

### 1. Build buddy-mlir with AME Support

Before you start trying the IR level examples, please make sure you have completed the [get started part](../../README.md).


### 2. Build and Test AME Examples

Navigate to the AME dialect examples directory:

```bash
cd buddy-mlir/examples/XTAMEDialect
```

#### Option A: Verify Assembly Generation (Quick Test)

Use this option to verify that AME operations are correctly lowered to RISC-V assembly. This uses the minimal `*.mlir` files.

**Generate Lowered MLIR:**

```bash
make ame-th-mmacc-w-b-lower      # integer operations
```

This generates `.mlir` containing the lowered representation.

**Generate LLVM IR:**

```bash
make ame-th-mmacc-w-b-translate
```

This generates `.ll` containing LLVM IR code.

**Generate Assembly:**

```bash
make ame-th-mmacc-w-b-asm
```

This generates `.s` containing RISC-V assembly. You can inspect this file to verify correct AME instruction generation.

**Clear generated files:**

```bash
make clean
```

#### Option B: Build Hardware Test Executables (Full Test)

An executable will be generated that can run on XuanTie-gcc hardware with print outputs for verification. Cross-compile the combined assembly file .s and the print file .c to generate an executable file.

**Build a single test executable:**

```bash
cd xuantie-gcc/bin

./riscv64-unknown-linux-gnu-gcc -static -march=rv64imafdc_xtheadmatrix03_xtheadmatrixmin_xtheadmmi8bi32b ~/test_dir/ame-th-mmacc-w-b.s ~/test_dir/run-ame-th-mmacc-w-b.c -o ~/test_dir/ame-th-mmacc-w-bexecutable
```

### Run on Hardware

**Test and verify on the xuantie-qemu emulator:**

```bash
cd xuantie-qemu/bin

./qemu-riscv64 ~/test_dir/ame-th-mmacc-w-b
```

## Reference

- xuantie-matrix-extension-0.5.0

---

### Operations Reference

This section provides detailed information about each AME operation type and how to use them in MLIR.

### Instruction Categories

AME instructions are categorized into two main types:

#### 1. Integer Matrix Multiply-Accumulate Instructions

These instructions perform signed or unsigned integer matrix operations:

| Instruction      | Operand A Type | Operand B Type | Accumulator Type | Description         |
| ---------------- | -------------- | -------------- | ---------------- | ------------------- |
| `th.mmacc.w.b`   | signed 8bit    | signed 8bit    | quad-widen       | signed × signed     |
| `th.mmaccu.w.b`  | unsigned 8bit  | unsigned 8bit  | quad-widen       | unsigned × unsigned |
| `th.mmaccus.w.b` | unsigned 8bit  | signed 8bit    | quad-widen       | unsigned × signed   |
| `th.mmaccsu.w.b` | signed  8bit   | unsigned 8bit  | quad-widen       | signed × unsigned   |

**Assembly Format**:

```assembly
th.mmacc.w.b md, ms2, ms1 #signed 8bit, output quad-widen
th.mmaccu.w.b md, ms2, ms1 #unsigned 8bit, output quad-widen
th.mmaccus.w.b md, ms2, ms1 #unsigned-signed 8bit, output quad-widen
th.mmaccsu.w.b md, ms2, ms1 #signed-unsigned 8bit, output quad-widen
```

**Register Constraints**:

- `md` (destination): Target register for result matrix C
- `ms1` (source 1): Input matrix A
- `ms2` (source 2): Input matrix B

#### 2. Floating-Point Matrix Multiply-Accumulate Instructions

This instruction performs floating-point matrix operations:

| Instruction | Operand A Type          | Operand B Type          | Accumulator Type                 | Description                    |
| ----------- | ----------------------- | ----------------------- | -------------------------------- | ------------------------------ |
| `th.mfmacc` | fp8/fp16/bf16/fp32/fp64 | fp8/fp16/bf16/fp32/fp64 | No widen/double-widen/quad-widen | Floating-point matrix multiply |

**Assembly Format**:

```assembly
th.mfmacc.h md, ms2, ms1 # 16-bit float point(fp16)
th.mfmacc.bf16 md, ms2, ms1 # 16-bit float point(bf16)
th.mfmacc.s md, ms2, ms1 # 32-bit float point
th.mfmacc.d md, ms2, ms1 # 64-bit float point
th.mfmacc.h.e4m3 md, ms2, ms1 # 8-bit float point, output double-widen
th.mfmacc.h.e5m2 md, ms2, ms1 # 8-bit float point, output double-widen
th.mfmacc.bf16.e4m3 md, ms2, ms1 # 8-bit float point, output double-widen
th.mfmacc.bf16.e5m2 md, ms2, ms1 # 8-bit float point, output double-widen
th.mfmacc.s.h md, ms2, ms1 # 16-bit float point, output double-widen
th.mfmacc.s.bf16 md, ms2, ms1 # 16-bit float point, output double-widen
th.mfmacc.d.s md, ms2, ms1 # 32-bit float point, output double-widen
th.mfmacc.s.e4m3 md, ms2, ms1 # 8-bit float point, output quad-widen
th.mfmacc.s.e5m2 md, ms2, ms1 # 8-bit float point, output quad-widen
```

**Register Constraints**:

- `md` (destination): Target register for result matrix C
- `ms1` (source 1): Input matrix A
- `ms2` (source 2): Input matrix B

### Example MLIR Code

Complete example showing AME operations in MLIR:

```mlir
// RUN: buddy-opt %s --lower-xt-ame | FileCheck %s

module {

  func.func private @print_C(i32, i32, i32, i32)

  func.func @main() -> i32 {

    %c_ptr = memref.alloc() : memref<4x4xi32>    // result matrix C
    %a_ptr = memref.alloc() : memref<4x4xi8>     // matrix A
    %b_ptr = memref.alloc() : memref<4x4xi8>     // matrix B

    %stride_a = arith.constant 4 : i64               // A's row stride (4 bytes per row)
    %stride_b = arith.constant 4 : i64               // B's row stride (4 bytes per row)
    %stride_c = arith.constant 16 : i64              // C's row stride (16 bytes per row, since each element is 4 bytes)

    //index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    //initialize A and B with some values (for testing)
    %v0 = arith.constant 0 : i8
    %v1 = arith.constant 1 : i8
    %v2 = arith.constant 2 : i8
    %v3 = arith.constant 3 : i8
    %v4 = arith.constant 4 : i8
    %v5 = arith.constant 5 : i8
    %v6 = arith.constant 6 : i8
    %v7 = arith.constant 7 : i8
    %v8 = arith.constant 8 : i8
    %v9 = arith.constant 9 : i8
    %v10 = arith.constant 10 : i8
    %v11 = arith.constant 11 : i8
    %v12 = arith.constant 12 : i8
    %v13 = arith.constant 13 : i8
    %v14 = arith.constant 14 : i8
    %v15 = arith.constant 15 : i8
    %v16 = arith.constant 16 : i8

    memref.store %v1, %a_ptr[%i0, %i0] : memref<4x4xi8>
    memref.store %v2, %a_ptr[%i0, %i1] : memref<4x4xi8>
    memref.store %v3, %a_ptr[%i0, %i2] : memref<4x4xi8>
    memref.store %v4, %a_ptr[%i0, %i3] : memref<4x4xi8>
    memref.store %v5, %a_ptr[%i1, %i0] : memref<4x4xi8>
    memref.store %v6, %a_ptr[%i1, %i1] : memref<4x4xi8>
    memref.store %v7, %a_ptr[%i1, %i2] : memref<4x4xi8>
    memref.store %v8, %a_ptr[%i1, %i3] : memref<4x4xi8>
    memref.store %v9, %a_ptr[%i2, %i0] : memref<4x4xi8>
    memref.store %v10, %a_ptr[%i2, %i1] : memref<4x4xi8>
    memref.store %v11, %a_ptr[%i2, %i2] : memref<4x4xi8>
    memref.store %v12, %a_ptr[%i2, %i3] : memref<4x4xi8>
    memref.store %v13, %a_ptr[%i3, %i0] : memref<4x4xi8>
    memref.store %v14, %a_ptr[%i3, %i1] : memref<4x4xi8>
    memref.store %v15, %a_ptr[%i3, %i2] : memref<4x4xi8>
    memref.store %v16, %a_ptr[%i3, %i3] : memref<4x4xi8>

    memref.store %v1, %b_ptr[%i0, %i0] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i0, %i1] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i0, %i2] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i0, %i3] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i1, %i0] : memref<4x4xi8>
    memref.store %v1, %b_ptr[%i1, %i1] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i1, %i2] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i1, %i3] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i2, %i0] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i2, %i1] : memref<4x4xi8>
    memref.store %v1, %b_ptr[%i2, %i2] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i2, %i3] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i3, %i0] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i3, %i1] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i3, %i2] : memref<4x4xi8>
    memref.store %v1, %b_ptr[%i3, %i3] : memref<4x4xi8>

    // Step 1: Configure tile dimensions
    // For a simple 4x4 tile operation
    xt_ame.th.mcfgmi 4          // mtilem = 4 (rows of C and A)
    xt_ame.th.mcfgni 4          // mtilen = 4 (cols of C and B)
    xt_ame.th.mcfgki 4          // mtilek = 4 (cols of A, rows of B)

    // Step 2: Zero the accumulation register (acc register 0)
    xt_ame.th.mzero 0

    // Step 3: Load matrix A to tile register 0 (shape: mtilem x mtilek = 4x4)
    xt_ame.th.mlde8 1, %stride_a, %a_ptr: memref<4x4xi8>

    // Step 4: Load transposed matrix B to tile register 1 (shape: mtilen x mtilek = 4x4)
    xt_ame.th.mldte8 2, %stride_b, %b_ptr: memref<4x4xi8>

    // Step 5: Execute matrix multiply: acc0 = acc0 + tile0 x tile1
    xt_ame.th.mmacc.w.b 0, 2, 1

    xt_ame.th.mcfgki 16         // mtilek = 16
    // Step 6: Store result from accumulator 0 to memory
    xt_ame.th.mste32 0, %stride_c, %c_ptr: memref<4x4xi32>

    //row 0
    %val_c00 = memref.load %c_ptr[%i0, %i0] : memref<4x4xi32>
    %val_c01 = memref.load %c_ptr[%i0, %i1] : memref<4x4xi32>
    %val_c02 = memref.load %c_ptr[%i0, %i2] : memref<4x4xi32>
    %val_c03 = memref.load %c_ptr[%i0, %i3] : memref<4x4xi32>
    call @print_C(%val_c00, %val_c01, %val_c02, %val_c03) : (i32, i32, i32, i32) -> ()

    //row 1
    %val_c10 = memref.load %c_ptr[%i1, %i0] : memref<4x4xi32>
    %val_c11 = memref.load %c_ptr[%i1, %i1] : memref<4x4xi32>
    %val_c12 = memref.load %c_ptr[%i1, %i2] : memref<4x4xi32>
    %val_c13 = memref.load %c_ptr[%i1, %i3] : memref<4x4xi32>
    call @print_C(%val_c10, %val_c11, %val_c12, %val_c13) : (i32, i32, i32, i32) -> ()

    //row 2
    %val_c20 = memref.load %c_ptr[%i2, %i0] : memref<4x4xi32>
    %val_c21 = memref.load %c_ptr[%i2, %i1] : memref<4x4xi32>
    %val_c22 = memref.load %c_ptr[%i2, %i2] : memref<4x4xi32>
    %val_c23 = memref.load %c_ptr[%i2, %i3] : memref<4x4xi32>
    call @print_C(%val_c20, %val_c21, %val_c22, %val_c23) : (i32, i32, i32, i32) -> ()

    //row 3
    %val_c30 = memref.load %c_ptr[%i3, %i0] : memref<4x4xi32>
    %val_c31 = memref.load %c_ptr[%i3, %i1] : memref<4x4xi32>
    %val_c32 = memref.load %c_ptr[%i3, %i2] : memref<4x4xi32>
    %val_c33 = memref.load %c_ptr[%i3, %i3] : memref<4x4xi32>
    call @print_C(%val_c30, %val_c31, %val_c32, %val_c33) : (i32, i32, i32, i32) -> ()

    memref.dealloc %c_ptr : memref<4x4xi32>
    memref.dealloc %a_ptr : memref<4x4xi8>
    memref.dealloc %b_ptr : memref<4x4xi8>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
  // For now, we only test the tile-level operations which map directly
  // to LLVM intrinsics.
}

// Expected lowering for tile-based operations:
// CHECK-LABEL: func.func @main
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgmi
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgni
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgki
// CHECK: llvm.call @llvm.riscv.buddy.th.mzero
// CHECK: llvm.call @llvm.riscv.buddy.th.mlde8
// CHECK: llvm.call @llvm.riscv.buddy.th.mldte8
// CHECK: llvm.call @llvm.riscv.buddy.th.mmacc.w.b
// CHECK: llvm.call @llvm.riscv.buddy.th.mste32

```

## Note：

### Unsigned matrix multiplication

We tested larger matrix sizes for unsigned matrix multiplication.

A perfect challenge spanning 'from single instructions to real algorithms'! Since the maximum single computation ability of the hardware (at RLEN=512) is locked at $16 times 64$ and $64 times 16$ resulting in $16 times 16$, for a large matrix with $M=64, N=64, K=256$, we need to cut it into small blocks like slicing tofu (Tiling), and then use three nested loops to feed these small blocks to the matrix units of the hardware in batches.

Let's break down the mathematical model of this task:

- Row direction (M): $64 / 16 = 4$ blocks.
- Column direction (N): $64 / 16 = 4$ blocks.
- Depth direction (K): $256 / 64 = 4$ blocks.

That is, you need $4 \times 4 = 16$ times $16 \times 16$ C matrix sub-blocks to form the final result.

Each C matrix sub-block requires $4$ multiply-accumulate operations along the K dimension.

In total, 64 times physical matrix multiplication instructions need to be executed.

```mlir
    // ==========================================
    // 5. Blocked Matrix Multiplication (Tiling)
    // ==========================================
    // Outer M loop (0, 16, 32, 48)
    scf.for %i = %c0 to %c64 step %step_m {

      // Mid-level N loop (0, 16, 32, 48)
      scf.for %j = %c0 to %c64 step %step_n {

        // Before calculating a new 16x16 block of C, the accumulator must be cleared
        xt_ame.th.mzero 0

        // The inner K loop accumulates (0, 64, 128, 192)
        scf.for %k = %c0 to %c256 step %step_k {

          // Extract subviews for the current tiles of A and B
          %sub_a = memref.subview %a_ptr[%i, %k] [16, 64] [1, 1]
                   : memref<64x256xi8> to memref<16x64xi8, strided<[256, 1], offset: ?>>

          %sub_b = memref.subview %b_ptr[%k, %j] [64, 16] [1, 1]
                   : memref<256x64xi8> to memref<64x16xi8, strided<[64, 1], offset: ?>>

          // Load and compute
          xt_ame.th.mlde8 1, %stride_a, %sub_a : memref<16x64xi8, strided<[256, 1], offset: ?>>
          xt_ame.th.mldte8 2, %stride_b, %sub_b : memref<64x16xi8, strided<[64, 1], offset: ?>>
          xt_ame.th.mmaccu.w.b 0, 2, 1
        }

        // Store back to C matrix memory
        %sub_c = memref.subview %c_ptr[%i, %j] [16, 16] [1, 1]
                 : memref<64x64xi32> to memref<16x16xi32, strided<[64, 1], offset: ?>>
        xt_ame.th.mste32 0, %stride_c, %sub_c : memref<16x16xi32, strided<[64, 1], offset: ?>>

      }
    }
```

This is the standard operation for handling large array slices in MLIR. It does not copy memory but generates a 'fat pointer (Descriptor)' with an offset.

**question 1：**The default commands in the Makefile are not enough, so we directly type the full downgrade compilation command in the terminal. We need to explicitly include the pass for handling memory slices (-expand-strided-metadata).

**question 2：**When testing on the xuantie-qemu emulator, some specific commands must be added.

```
./qemu-riscv64 ~/test_dir/ame-th-mmaccu-w-b
```

The original basic command will trigger an illegal instruction error.

```
./qemu-riscv64 -cpu max,rlen=512 ~/test_dir/ame-th-mmaccu-w-b
```

It is necessary to add the mandatory command `-cpu max,rlen=512` to achieve optimal performance of the emulator.
