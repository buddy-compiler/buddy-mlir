// RUN: buddy-opt %s --lower-ame | FileCheck %s

// Test for AME mqma.b.mm operation (int8 quad-widen matrix multiply)
// This performs: C = C + A * B where A and B are int8, C is int32

module {
  func.func @test_mqma_b_mm(%C: memref<4x4xi32>, %A: memref<4x8xi8>, %B: memref<8x4xi8>) {
    // int8 quad-widen matrix multiply: C[4x4] += A[4x8] × B[8x4]
    ame.mqma.b.mm %C, %A, %B : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
    return
  }

  // CHECK-LABEL: func.func @test_mqma_b_mm
  // CHECK: llvm.call @llvm.riscv.buddy.mqma.b.mm
}
