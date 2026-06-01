// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s

// ===========================================================================
// Complete Matrix Multiplication Demo using RISC-V Matrix Extension (BOSC AME)
// ===========================================================================
//
// This demo shows the complete flow of matrix multiplication:
// 1. Configure tile dimensions (msettilemi, msettileni, msettileki)
// 2. Zero accumulator (mzero)
// 3. Load matrix tiles (mlae32.m, mlbe32.m)
// 4. Execute matrix multiply (mma.w.mm)
// 5. Store result (msce32.m)
//
// Matrix dimensions: C[M×N] = A[M×K] × B[K×N]
// Tile dimensions are configured via msettilem/msettilen/msettilek
//
// ===========================================================================

module {

  func.func private @print_C(i32, i32, i32, i32)
  // Demo: int32 tile-based matrix multiplication
  // Uses tile register operations (hardware-level abstraction)
  func.func @main() -> i32 {

    %c_ptr = memref.alloc() : memref<4x4xi32>    // result matrix C
    %a_ptr = memref.alloc() : memref<4x4xi32>     // matrix A
    %b_ptr = memref.alloc() : memref<4x4xi32>     // matrix B

    %stride_a = arith.constant 4 : i64           // row stride for A
    %stride_b = arith.constant 4 : i64           // row stride for B
    %stride_c = arith.constant 4 : i64           // row stride for C

    //index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    //initialize A and B with some values (for testing)
    %v0 = arith.constant 0 : i32
    %v1 = arith.constant 1 : i32
    %v2 = arith.constant 2 : i32
    %v3 = arith.constant 3 : i32
    %v4 = arith.constant 4 : i32
    %v5 = arith.constant 5 : i32
    %v6 = arith.constant 6 : i32
    %v7 = arith.constant 7 : i32
    %v8 = arith.constant 8 : i32
    %v9 = arith.constant 9 : i32
    %v10 = arith.constant 10 : i32
    %v11 = arith.constant 11 : i32
    %v12 = arith.constant 12 : i32
    %v13 = arith.constant 13 : i32
    %v14 = arith.constant 14 : i32
    %v15 = arith.constant 15 : i32
    %v16 = arith.constant 16 : i32

    memref.store %v1, %a_ptr[%i0, %i0] : memref<4x4xi32>
    memref.store %v2, %a_ptr[%i0, %i1] : memref<4x4xi32>
    memref.store %v3, %a_ptr[%i0, %i2] : memref<4x4xi32>
    memref.store %v4, %a_ptr[%i0, %i3] : memref<4x4xi32>
    memref.store %v5, %a_ptr[%i1, %i0] : memref<4x4xi32>
    memref.store %v6, %a_ptr[%i1, %i1] : memref<4x4xi32>
    memref.store %v7, %a_ptr[%i1, %i2] : memref<4x4xi32>
    memref.store %v8, %a_ptr[%i1, %i3] : memref<4x4xi32>
    memref.store %v9, %a_ptr[%i2, %i0] : memref<4x4xi32>
    memref.store %v10, %a_ptr[%i2, %i1] : memref<4x4xi32>
    memref.store %v11, %a_ptr[%i2, %i2] : memref<4x4xi32>
    memref.store %v12, %a_ptr[%i2, %i3] : memref<4x4xi32>
    memref.store %v13, %a_ptr[%i3, %i0] : memref<4x4xi32>
    memref.store %v14, %a_ptr[%i3, %i1] : memref<4x4xi32>
    memref.store %v15, %a_ptr[%i3, %i2] : memref<4x4xi32>
    memref.store %v16, %a_ptr[%i3, %i3] : memref<4x4xi32>

    memref.store %v1, %b_ptr[%i0, %i0] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i0, %i1] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i0, %i2] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i0, %i3] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i1, %i0] : memref<4x4xi32>
    memref.store %v1, %b_ptr[%i1, %i1] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i1, %i2] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i1, %i3] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i2, %i0] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i2, %i1] : memref<4x4xi32>
    memref.store %v1, %b_ptr[%i2, %i2] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i2, %i3] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i3, %i0] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i3, %i1] : memref<4x4xi32>
    memref.store %v0, %b_ptr[%i3, %i2] : memref<4x4xi32>
    memref.store %v1, %b_ptr[%i3, %i3] : memref<4x4xi32>

    // Step 1: Configure tile dimensions
    // For a simple 4x4 tile operation
    %rd_m = bosc_ame.msettilemi 4 : i64          // mtilem = 4 (rows of C and A)
    %rd_n = bosc_ame.msettileni 4 : i64          // mtilen = 4 (cols of C and B)
    %rd_k = bosc_ame.msettileki 4 : i64          // mtilek = 4 (cols of A, rows of B)

    // Step 2: Zero the accumulation register (tile register 0)
    bosc_ame.mzero 0

    // Step 3: Load matrix A to tile register 0 (shape: mtilem x mtilek = 4x4)
    bosc_ame.mlae32.m 0, %a_ptr, %stride_a : memref<4x4xi32>

    // Step 4: Load matrix B to tile register 1 (shape: mtilek x mtilen = 4x4)
    bosc_ame.mlbe32.m 1, %b_ptr, %stride_b : memref<4x4xi32>

    // Step 5: Execute matrix multiply: acc0 = acc0 + tile0 x tile1
    bosc_ame.mma.w.mm 0, 0, 1

    // Step 6: Store result from accumulator 0 to memory
    bosc_ame.msce32.m 0, %c_ptr, %stride_c : memref<4x4xi32>

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
    memref.dealloc %a_ptr : memref<4x4xi32>
    memref.dealloc %b_ptr : memref<4x4xi32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }

  // NOTE: High-level mma.w.mm operation (memref abstraction) requires
  // additional lowering pass to convert memref to tile operations.
  // For now, we only test the tile-level operations which map directly
  // to LLVM intrinsics.
}

// Expected lowering for tile-based operations:
// CHECK-LABEL: func.func @main
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettilemi
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettileni
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettileki
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mzero
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mlae32.m
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mlbe32.m
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mma.w.mm
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msce32.m
