# SpacemiT IME Dialect

This document provides a comprehensive guide for using the IME (Integrated Matrix Extension) dialect in buddy-mlir.

## Table of Contents

1. [File Structure](#file-structure)
2. [Introduction](#introduction)
3. [Quick Start](#quick-start)
4. [Reference](#reference)

### File Structure

The IMEDialect example folder contains the following key files:

**MLIR Source Files:**
- **`vmadot.mlir`, `vmadotu.mlir`, `vmadotsu.mlir`, `vmadotus.mlir`, `vfmadot.mlir`**: Minimal MLIR files demonstrating each basic IME operation. Use these to verify assembly code generation.
- **`vmadot1.mlir`, `vmadot2.mlir`, `vmadot3.mlir`**: Fixed sliding-window integer operations with slide=1, 2, 3.
- **`vfmadot1.mlir`, `vfmadot2.mlir`, `vfmadot3.mlir`**: Fixed sliding-window floating-point operations with slide=1, 2, 3.
- **`vmadotn.mlir`, `vmadotnu.mlir`, `vmadotnsu.mlir`, `vmadotnus.mlir`, `vfmadotn.mlir`**: Minimal MLIR files for dynamic sliding-window operations with runtime slide parameter.
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
make vmadot-lower      # Basic integer operations
make vmadot1-lower     # Fixed sliding-window (slide=1)
make vmadotn-lower     # Dynamic sliding-window
make vfmadot-lower     # Floating-point
make vfmadot1-lower    # Floating-point fixed sliding-window
make vfmadotn-lower    # Floating-point dynamic sliding-window
```
This generates `log.mlir` containing the lowered representation.

**Generate LLVM IR:**
```bash
make vmadot-translate
make vmadot1-translate
make vmadotn-translate
# ... and similarly for other variants
```
This generates `log.ll` containing LLVM IR code.

**Generate Assembly:**
```bash
make vmadot-asm        # Basic: vmadot, vmadotu, vmadotsu, vmadotus
make vmadot1-asm       # Fixed slide: vmadot1, vmadot2, vmadot3
make vmadotn-asm       # Dynamic slide: vmadotn, vmadotnu, vmadotnsu, vmadotnus
make vfmadot-asm       # Float basic
make vfmadot1-asm      # Float fixed slide: vfmadot1, vfmadot2, vfmadot3
make vfmadotn-asm      # Float dynamic slide
```
This generates `log.s` containing RISC-V assembly. You can inspect this file to verify correct IME instruction generation.

**Available Make Targets Summary:**

| Category | Instructions | Make Targets |
|----------|-------------|--------------|
| Basic Integer | vmadot, vmadotu, vmadotsu, vmadotus | `vmadot{,u,su,us}-{lower,translate,asm,run}` |
| Fixed Slide Integer (signed) | vmadot1, vmadot2, vmadot3 | `vmadot{1,2,3}-{lower,translate,asm,run}` |
| Fixed Slide Integer (unsigned) | vmadot1u, vmadot2u, vmadot3u | `vmadot{1,2,3}u-{lower,translate,asm}` |
| Fixed Slide Integer (signed×unsigned) | vmadot1su, vmadot2su, vmadot3su | `vmadot{1,2,3}su-{lower,translate,asm}` |
| Fixed Slide Integer (unsigned×signed) | vmadot1us, vmadot2us, vmadot3us | `vmadot{1,2,3}us-{lower,translate,asm}` |
| Dynamic Slide Integer | vmadotn, vmadotnu, vmadotnsu, vmadotnus | `vmadotn{,u,su,us}-{lower,translate,asm}` |
| Basic Float | vfmadot | `vfmadot-{lower,translate,asm,run}` |
| Fixed Slide Float | vfmadot1, vfmadot2, vfmadot3 | `vfmadot{1,2,3}-{lower,translate,asm,run}` |
| Dynamic Slide Float | vfmadotn | `vfmadotn-{lower,translate,asm,run}` |

#### Option B: Build Hardware Test Executables (Full Test)

Use this option to build executables that can run on SpacemiT hardware with printed output for verification. This uses the `*_print_test.mlir` files which include matrix printing functionality.

**Build a single test executable:**
```bash
export PATH=$PWD/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin:$PATH
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

> **Toolchain Compatibility Note**: The SpacemiT toolchain (v1.1.2) currently only supports basic IME instructions (`vmadot`, `vmadotu`, `vmadotsu`, `vmadotus`). The following instructions will produce "unrecognized opcode" errors when assembled with SpacemiT's GNU as:
> - Fixed sliding-window: `vmadot1`, `vmadot2`, `vmadot3`, `vfmadot1`, `vfmadot2`, `vfmadot3`
> - Dynamic sliding-window: `vmadotn`, `vmadotnu`, `vmadotnsu`, `vmadotnus`, `vfmadotn`
> - Floating-point basic: `vfmadot`
>
> For these unsupported instructions, you can still verify assembly generation using the `-asm` targets, but hardware testing requires SpacemiT to update their toolchain.

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

**Current Hardware Test Status:**

| Instruction | Assembly Gen | Hardware Test | Notes |
|-------------|-------------|---------------|-------|
| vmadot | ✅ | ✅ | Fully working |
| vmadotu | ✅ | ✅ | Fully working |
| vmadotsu | ✅ | ✅ | Fully working |
| vmadotus | ✅ | ✅ | Fully working |
| vmadot1/2/3 | ✅ | ⏳ | Waiting for funct7 fix |
| vmadotn/nu/nsu/nus | ✅ | ⏳ | Waiting for toolchain support |
| vfmadot | ✅ | ⏳ | Waiting for toolchain support |
| vfmadot1/2/3 | ✅ | ⏳ | Waiting for toolchain support |
| vfmadotn | ✅ | ⏳ | Waiting for toolchain support |



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

#### 3. Fixed Sliding-Window Instructions (vmadot1/2/3, vfmadot1/2/3)

These instructions have a **fixed slide amount encoded in the instruction**. They read from VS1 and VS1+1 (64 elements for int8, forming a 2M×K matrix), then slide by a fixed number of rows (1, 2, or 3) to select an M×K submatrix.

**Integer Fixed Sliding-Window Instructions:**

| Category | Instructions | Operand A Type | Operand B Type | Accumulator Type | Description |
|----------|-------------|---|---|---|---|
| slide-1 | `vmadot1` | int8 | int8 | int32 | signed × signed |
|         | `vmadot1u` | uint8 | uint8 | int32 | unsigned × unsigned |
|         | `vmadot1su` | int8 | uint8 | int32 | signed × unsigned |
|         | `vmadot1us` | uint8 | int8 | int32 | unsigned × signed |
| slide-2 | `vmadot2` | int8 | int8 | int32 | signed × signed |
|         | `vmadot2u` | uint8 | uint8 | int32 | unsigned × unsigned |
|         | `vmadot2su` | int8 | uint8 | int32 | signed × unsigned |
|         | `vmadot2us` | uint8 | int8 | int32 | unsigned × signed |
| slide-3 | `vmadot3` | int8 | int8 | int32 | signed × signed |
|         | `vmadot3u` | uint8 | uint8 | int32 | unsigned × unsigned |
|         | `vmadot3su` | int8 | uint8 | int32 | signed × unsigned |
|         | `vmadot3us` | uint8 | int8 | int32 | unsigned × signed |

**Floating-Point Fixed Sliding-Window Instructions:**

| Instruction | Operand A Type | Operand B Type | Accumulator Type | Description |
|-------------|---|---|---|---|
| `vfmadot1` | fp16 | fp16 | fp16 | floating-point, slide=1 |
| `vfmadot2` | fp16 | fp16 | fp16 | floating-point, slide=2 |
| `vfmadot3` | fp16 | fp16 | fp16 | floating-point, slide=3 |

**Assembly Format**:
```assembly
# Integer slide-1
vmadot1   vd, vs1, vs2    # vd(C) += slide(vs1, 1)(A) × vs2(B) - signed × signed
vmadot1u  vd, vs1, vs2    # unsigned × unsigned
vmadot1su vd, vs1, vs2    # signed × unsigned
vmadot1us vd, vs1, vs2    # unsigned × signed

# Integer slide-2
vmadot2   vd, vs1, vs2
vmadot2u  vd, vs1, vs2
vmadot2su vd, vs1, vs2
vmadot2us vd, vs1, vs2

# Integer slide-3
vmadot3   vd, vs1, vs2
vmadot3u  vd, vs1, vs2
vmadot3su vd, vs1, vs2
vmadot3us vd, vs1, vs2

# Floating-point
vfmadot1 vd, vs1, vs2
vfmadot2 vd, vs1, vs2
vfmadot3 vd, vs1, vs2
```

**Register Constraints**:
- `vd` (destination): Target register for result matrix C
- `vs1` (source 1): Input matrix A (reads VS1 and VS1+1, 64 elements for sliding)
- `vs2` (source 2): Input matrix B

> **Note**: The `vmadot1/2/3` instructions currently have a funct7 encoding issue (see [SpacemiT Issue #2](https://github.com/user/repo/issues/2)). Assembly generation works, but machine code may be incorrect until resolved.

#### 4. Dynamic Sliding-Window Instructions (vmadotn/vfmadotn)

These instructions support a **dynamic slide parameter** passed at runtime, allowing flexible row selection from the source matrix. The sliding window reads from VS1 and VS1+1 (64 elements for int8, forming a 2M×K matrix), then slides by n rows to select an M×K submatrix.

| Instruction | Operand A Type | Operand B Type | Accumulator Type | Description |
|-------------|---|---|---|---|
| `vmadotn` | int8 | int8 | int32 | signed × signed with dynamic slide |
| `vmadotnu` | uint8 | uint8 | int32 | unsigned × unsigned with dynamic slide |
| `vmadotnsu` | int8 | uint8 | int32 | signed × unsigned with dynamic slide |
| `vmadotnus` | uint8 | int8 | int32 | unsigned × signed with dynamic slide |
| `vfmadotn` | fp16 | fp16 | fp16 | floating-point with dynamic slide |

**Assembly Format**:
```assembly
vmadotn   vd, vs1, vs2, rs1    # vd(C) += slide(vs1, rs1)(A) × vs2(B)
vmadotnu  vd, vs1, vs2, rs1
vmadotnsu vd, vs1, vs2, rs1
vmadotnus vd, vs1, vs2, rs1
vfmadotn  vd, vs1, vs2, rs1
```

**Register Constraints**:
- `vd` (destination): Target register for result matrix C
- `vs1` (source 1): Input matrix A (reads VS1 and VS1+1, 64 elements for sliding)
- `vs2` (source 2): Input matrix B
- `rs1` (scalar): Slide amount (0-3 for int8 with VLEN=256)

**Sliding Window Mechanism**:
```
For VLEN=256, int8:
- VS1 loads 64 elements (8 rows × 8 cols)
- slide=0: use rows [0,1,2,3]
- slide=1: use rows [1,2,3,4]
- slide=2: use rows [2,3,4,5]
- slide=3: use rows [3,4,5,6]
```

### MLIR Operation Syntax

All IME operations in MLIR follow this pattern:

```mlir
// Basic operations (without slide)
ime.vmadot %accumulator, %matrix_a, %matrix_b : memref<...>, memref<...>, memref<...>

// Dynamic sliding-window operations (with slide parameter)
ime.vmadotn %accumulator, %matrix_a, %matrix_b, %slide : memref<...>, memref<...>, memref<...>, i64
```

Where:
- `%accumulator`: Destination memref (2D, element type matches result type)
- `%matrix_a`: Left operand matrix A memref (2D)
- `%matrix_b`: Right operand matrix B memref (2D)
- `%slide`: (for vmadotn variants) Slide amount as i64 scalar

### Example MLIR Code

Complete example showing IME operations in MLIR:

```mlir
// Basic integer matrix multiply-accumulate
func.func @vmadot_example(%arg0: memref<4x4xi32>, %arg1: memref<4x8xi8>, %arg2: memref<8x4xi8>) {
  // Perform matrix multiply-accumulate: arg0 += arg1 × arg2
  ime.vmadot %arg0, %arg1, %arg2 : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  return
}

// Unsigned integer version
func.func @vmadotu_example(%arg0: memref<4x4xi32>, %arg1: memref<4x8xui8>, %arg2: memref<8x4xui8>) {
  ime.vmadotu %arg0, %arg1, %arg2 : memref<4x4xi32>, memref<4x8xui8>, memref<8x4xui8>
  return
}

// Floating-point version
func.func @vfmadot_example(%arg0: memref<4x4xf16>, %arg1: memref<4x4xf16>, %arg2: memref<4x4xf16>) {
  ime.vfmadot %arg0, %arg1, %arg2 : memref<4x4xf16>, memref<4x4xf16>, memref<4x4xf16>
  return
}

// Fixed sliding-window: A is 8x8 (2M×K), slide=1 selects rows [1,2,3,4]
// Signed × Signed variants
func.func @vmadot1_example(%arg0: memref<4x4xi32>, %arg1: memref<8x8xi8>, %arg2: memref<8x4xi8>) {
  ime.vmadot1 %arg0, %arg1, %arg2 : memref<4x4xi32>, memref<8x8xi8>, memref<8x4xi8>
  return
}

// Unsigned × Unsigned variants
func.func @vmadot1u_example(%arg0: memref<4x4xi32>, %arg1: memref<8x8xui8>, %arg2: memref<8x4xui8>) {
  ime.vmadot1u %arg0, %arg1, %arg2 : memref<4x4xi32>, memref<8x8xui8>, memref<8x4xui8>
  return
}

// Signed × Unsigned variants
func.func @vmadot1su_example(%arg0: memref<4x4xi32>, %arg1: memref<8x8xi8>, %arg2: memref<8x4xui8>) {
  ime.vmadot1su %arg0, %arg1, %arg2 : memref<4x4xi32>, memref<8x8xi8>, memref<8x4xui8>
  return
}

// Unsigned × Signed variants
func.func @vmadot1us_example(%arg0: memref<4x4xi32>, %arg1: memref<8x8xui8>, %arg2: memref<8x4xi8>) {
  ime.vmadot1us %arg0, %arg1, %arg2 : memref<4x4xi32>, memref<8x8xui8>, memref<8x4xi8>
  return
}

// Fixed sliding-window floating-point
func.func @vfmadot1_example(%arg0: memref<4x4xf16>, %arg1: memref<8x4xf16>, %arg2: memref<4x4xf16>) {
  ime.vfmadot1 %arg0, %arg1, %arg2 : memref<4x4xf16>, memref<8x4xf16>, memref<4x4xf16>
  return
}

// Dynamic sliding-window: slide amount passed at runtime
func.func @vmadotn_example(%arg0: memref<4x4xi32>, %arg1: memref<8x8xi8>, %arg2: memref<8x4xi8>, %slide: i64) {
  ime.vmadotn %arg0, %arg1, %arg2, %slide : memref<4x4xi32>, memref<8x8xi8>, memref<8x4xi8>, i64
  return
}

// Dynamic sliding-window floating-point
func.func @vfmadotn_example(%arg0: memref<4x4xf16>, %arg1: memref<8x4xf16>, %arg2: memref<4x4xf16>, %slide: i64) {
  ime.vfmadotn %arg0, %arg1, %arg2, %slide : memref<4x4xf16>, memref<8x4xf16>, memref<4x4xf16>, i64
  return
}
```
