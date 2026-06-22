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
  // Demo: int32 tile-based matrix multiplication
  // Uses tile register operations (hardware-level abstraction)
  func.func @mma_w_mm_demo(%c_ptr: memref<?x?xi32>,
                                %a_ptr: memref<?x?xi32>,
                                %b_ptr: memref<?x?xi32>,
                                %stride_a: i64,
                                %stride_b: i64,
                                %stride_c: i64) {

    // Step 1: Configure tile dimensions
    // For a simple 4x4 tile operation
    %rd_m = bosc_ame.msettilemi 4 : i64          // mtilem = 4 (rows of C and A)
    %rd_n = bosc_ame.msettileni 4 : i64          // mtilen = 4 (cols of C and B)
    %rd_k = bosc_ame.msettileki 8 : i64          // mtilek = 8 (cols of A, rows of B)

    // Step 2: Zero the accumulation register (tile register 0)
    bosc_ame.mzero 0

    // Step 3: Load matrix A to tile register 0 (shape: mtilem x mtilek = 4x8)
    bosc_ame.mlae32.m 0, %a_ptr, %stride_a : memref<?x?xi32>

    // Step 4: Load matrix B to tile register 1 (shape: mtilek x mtilen = 8x4)
    bosc_ame.mlbe32.m 1, %b_ptr, %stride_b : memref<?x?xi32>

    // Step 5: Execute matrix multiply: acc0 = acc0 + tile0 x tile1
    bosc_ame.mma.w.mm 0, 0, 1

    // Step 6: Store result from accumulator 0 to memory
    bosc_ame.msce32.m 0, %c_ptr, %stride_c : memref<?x?xi32>

    return
  }

  // NOTE: High-level mma.w.mm operation (memref abstraction) requires
  // additional lowering pass to convert memref to tile operations.
  // For now, we only test the tile-level operations which map directly
  // to LLVM intrinsics.
}

// Expected lowering for tile-based operations:
// CHECK-LABEL: func.func @mma_w_mm_demo
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettilemi
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettileni
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msettileki
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mzero
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mlae32.m
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mlbe32.m
// CHECK: llvm.call @llvm.riscv.buddy.bosc.mma.w.mm
// CHECK: llvm.call @llvm.riscv.buddy.bosc.msce32.m
