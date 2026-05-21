// RUN: buddy-opt %s -lower-linalg-to-xtame | FileCheck %s
//
// This file tests the lowering of linalg.matmul to xtame.vmadot operations.
//

// CHECK-LABEL: func.func @matmul_i8_4x4x4
// CHECK: xt_ame.th.mcfgmi 4
// CHECK: xt_ame.th.mcfgni 4
// CHECK: xt_ame.th.mcfgki 4
func.func @matmul_i8_4x4x4(%A: memref<4x4xi8>, %B: memref<4x4xi8>, %C: memref<4x4xi32>) {
  linalg.matmul ins(%A, %B : memref<4x4xi8>, memref<4x4xi8>)
                outs(%C : memref<4x4xi32>)
  return
}
