# SpacemiT IME Dialect

This document provides a comprehensive guide for using the IME (Intelligent Matrix Extension) dialect in buddy-mlir.

## Table of Contents

1. [File Structure](#file-structure)
2. [Introduction](#introduction)
3. [Quick Start](#quick-start)
4. [Reference](#reference)

### File Structure

The IMEDialect example folder contains the following key files:

**MLIR Source Files:**
- **`vmadot.mlir`, `vmadotu.mlir`, `vmadotsu.mlir`, `vmadotus.mlir`, `vfmadot.mlir`**: Minimal MLIR files demonstrating each IME operation. Use these to verify assembly code generation.
- **`vmadot_print_test.mlir`, `vmadotu_print_test.mlir`, etc.**: Extended test MLIR files with additional code for printing input/output matrices on hardware. Use these for hardware validation with visible results.

**Runtime and Build Files:**
- **`runtime_*.c`**: C runtime files providing `main()` entry point, matrix initialization, and result verification for hardware testing.
- **`makefile`**: Build automation for generating lowered MLIR, LLVM IR, and assembly code.
- **`build_all_tests.sh`**: Convenience script to build all hardware test executables at once.

## Introduction

The IME dialect provides MLIR operations that map to SpacemiT's Intelligent Matrix Extension for RISC-V. 

### Matrix Operations

All IME instructions perform the following core matrix multiply-accumulate operation:

```
C (M×N) += A (M×K) × B (K×N)
```

Where:
- **M, N, K**: Matrix dimensions (determined by VLEN configuration)
- **A (vs1)**: Left operand matrix (int8/int4/int16 or fp16/fp8/fp4/bfp16)
- **B (vs2)**: Right operand matrix (int8/int4/int16 or fp16/fp8/fp4/bfp16)
- **C (vd)**: Accumulator/result matrix (int32 for integer ops, fp16/bfp16 for float ops)

### Matrix Dimensions by Configuration

IME operates on fixed tile sizes determined by VLEN (Vector Register Length):

| VLEN | Data Type | M | K | N | Description |
|------|-----------|---|---|---|-------------|
| 256 | int8 | 4 | 8 | 4 | 4×8 × 8×4 → 4×4 |
| 256 | fp16 | 4 | 4 | 4 | 4×4 × 4×4 → 4×4 |
| 128 | int8 | 4 | 4 | 4 | 4×4 × 4×4 → 4×4 |
| 128 | fp16 | 4 | 2 | 4 | 4×2 × 2×4 → 4×4 |


## Quick Start

### Prerequisites

Before you start, ensure you have the following prerequisites installed:

```bash
# Download and setup SpacemiT cross-compiler toolchain
wget https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz
tar -xvf spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz
export PATH=$PWD/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin:$PATH
```

### 1. Build buddy-mlir with IME Support

Before you start trying the IR level examples, please make sure you have completed the [get started part](../../README.md).


### 2. Build and Test IME Examples

Navigate to the IME dialect examples directory:

```bash
cd buddy-mlir/examples/IMEDialect
```

#### Option A: Verify Assembly Generation (Quick Test)

Use this option to verify that IME operations are correctly lowered to RISC-V assembly. This uses the minimal `*.mlir` files.

**Generate Lowered MLIR:**
```bash
make vmadot-lower
```
This generates `log.mlir` containing the lowered representation.

**Generate LLVM IR:**
```bash
make vmadot-translate
```
This generates `log.ll` containing LLVM IR code.

**Generate Assembly:**
```bash
make vmadot-asm
```
This generates `log.s` containing RISC-V assembly. You can inspect this file to verify correct IME instruction generation.

#### Option B: Build Hardware Test Executables (Full Test)

Use this option to build executables that can run on SpacemiT hardware with printed output for verification. This uses the `*_print_test.mlir` files which include matrix printing functionality.

**Build a single test executable:**
```bash
make vmadot-run    # Generates vmadot.s and vmadot_test executable
make vmadotu-run   # Generates vmadotu.s and vmadotu_test executable
make vmadotsu-run  # Generates vmadotsu.s and vmadotsu_test executable
make vmadotus-run  # Generates vmadotus.s and vmadotus_test executable
```

**Build all test executables at once:**
```bash
./build_all_tests.sh
```

This will generate executable binaries: `vmadot_test`, `vmadotu_test`, `vmadotsu_test`, `vmadotus_test`

> **Note**: The `-run` targets require the SpacemiT cross-compiler (`riscv64-unknown-linux-gnu-gcc`) to be in your PATH.

---

## Run on Hardware

After compiling the test executables, you can run them on SpacemiT hardware:

**Step 1: Transfer files to hardware**

```bash
scp vmadot_test vmadotu_test vmadotsu_test vmadotus_test \
    user@spacemit-hardware:/path/to/test_dir/
```

**Step 2: Execute on hardware**

```bash
# SSH into the SpacemiT hardware
ssh user@spacemit-hardware

# Navigate to test directory and run tests
cd /path/to/test_dir/
./vmadot_test
./vmadotu_test
./vmadotsu_test
./vmadotus_test
```

Each test will output:
- Input matrices A and B
- Expected and computed results
- Verification status (PASS/FAIL)

**Note**: `vfmadot` floating-point instruction tests cannot yet be executed as standalone binaries and are available only for assembly generation and inspection.



## Reference
- [SpacemiT IME Extension Specification](https://github.com/spacemit-com/riscv-ime-extension-spec)

---

## Operations Reference

This section provides detailed information about each IME operation type and how to use them in MLIR.

### Instruction Categories

IME instructions are categorized into two main types:

#### 1. Integer Matrix Multiply-Accumulate Instructions

These instructions perform signed or unsigned integer matrix operations:

| Instruction | Operand A Type | Operand B Type | Accumulator Type | Description |
|-------------|---|---|---|---|
| `vmadot` | int4/int8/int16 | int4/int8/int16 | int32 | signed × signed |
| `vmadotu` | uint4/uint8/uint16 | uint4/uint8/uint16 | int32 | unsigned × unsigned |
| `vmadotsu` | int4/int8/int16 | uint4/uint8/uint16 | int32 | signed × unsigned |
| `vmadotus` | uint4/uint8/uint16 | int4/int8/int16 | int32 | unsigned × signed |

**Assembly Format**:
```assembly
vmadot   vd, vs1, vs2    # vd(C) += vs1(A) × vs2(B)
vmadotu  vd, vs1, vs2
vmadotsu vd, vs1, vs2
vmadotus vd, vs1, vs2
```

**Register Constraints**:
- `vd` (destination): Target register for result matrix C, **index must be even**
- `vs1` (source 1): Input matrix A
- `vs2` (source 2): Input matrix B
- Results are stored in two consecutive registers (vd and vd+1)

#### 2. Floating-Point Matrix Multiply-Accumulate Instructions

This instruction performs floating-point matrix operations:

| Instruction | Operand A Type | Operand B Type | Accumulator Type | Description |
|---|---|---|---|---|
| `vfmadot` | fp4/fp8/fp16/bfp16 | fp4/fp8/fp16/bfp16 | fp16/bfp16 | Floating-point matrix multiply |

**Assembly Format**:
```assembly
vfmadot vd, vs1, vs2    # vd(C) += vs1(A) × vs2(B)
```

**Register Constraints**:
- `vd` (destination): Target register for result matrix C
- `vs1` (source 1): Input matrix A
- `vs2` (source 2): Input matrix B
- Result is stored in a single register (different from integer instructions)

### MLIR Operation Syntax

All IME operations in MLIR follow this pattern:

```mlir
ime.vmadot %accumulator, %matrix_a, %matrix_b : memref<...>, memref<...>, memref<...>
```

Where:
- `%accumulator`: Destination memref (2D, element type matches result type)
- `%matrix_a`: Left operand matrix A memref (2D)
- `%matrix_b`: Right operand matrix B memref (2D)

### Example MLIR Code

Complete example showing IME operations in MLIR:

```mlir
func.func @vmadot_example(%arg0: memref<4x4xi32>, %arg1: memref<4x8xi8>, %arg2: memref<8x4xi8>) {
  // Perform matrix multiply-accumulate: arg0 += arg1 × arg2
  ime.vmadot %arg0, %arg1, %arg2 : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  return
}

func.func @vmadotu_example(%arg0: memref<4x4xi32>, %arg1: memref<4x8xui8>, %arg2: memref<8x4xui8>) {
  // Unsigned integer version
  ime.vmadotu %arg0, %arg1, %arg2 : memref<4x4xi32>, memref<4x8xui8>, memref<8x4xui8>
  return
}

func.func @vfmadot_example(%arg0: memref<4x4xf16>, %arg1: memref<4x4xf16>, %arg2: memref<4x4xf16>) {
  // Floating-point version
  ime.vfmadot %arg0, %arg1, %arg2 : memref<4x4xf16>, memref<4x4xf16>, memref<4x4xf16>
  return
}
```
