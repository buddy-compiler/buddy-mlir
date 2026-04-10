// RUN: buddy-opt %s --lower-ame | FileCheck %s

// Demo for mma.w.mm (int32 matrix multiply-accumulate)
// Performs: md = md + ms1 × ms2
// where ms1, ms2, and md are all int32 matrices.

module {
  func.func @mma_w_mm_demo(%md: memref<4x4xi32>,
                           %ms1: memref<4x8xi32>,
                           %ms2: memref<8x4xi32>) {
    // int32 matrix multiply: C[4x4] += A[4x8] × B[8x4]
    ame.mma.w.mm %md, %ms1, %ms2 : memref<4x4xi32>, memref<4x8xi32>, memref<8x4xi32>
    return
  }

  // CHECK-LABEL: func.func @mma_w_mm_demo
  // CHECK: llvm.call @llvm.riscv.buddy.mma.w.mm
}
