// RUN: buddy-opt %s --lower-ame | FileCheck %s

// Demo for mma.dw.mm (int64 matrix multiply-accumulate)
// Performs: md = md + ms1 × ms2
// where ms1, ms2, and md are all int64 matrices.
// This instruction is useful for high-precision computations.

module {
  func.func @mma_dw_mm_demo(%md: memref<4x4xi64>,
                            %ms1: memref<4x8xi64>,
                            %ms2: memref<8x4xi64>) {
    // int64 matrix multiply: C[4x4] += A[4x8] × B[8x4]
    ame.mma.dw.mm %md, %ms1, %ms2 : memref<4x4xi64>, memref<4x8xi64>, memref<8x4xi64>
    return
  }

  // CHECK-LABEL: func.func @mma_dw_mm_demo
  // CHECK: llvm.call @llvm.riscv.buddy.mma.dw.mm
}
