// RUN: buddy-opt %s -lower-linalg-to-boscame | FileCheck %s
//
// This file tests lowering a linalg.generic transpose pattern to BOSCAME.

#identity_2d = affine_map<(d0, d1) -> (d0, d1)>
#transpose_in = affine_map<(d0, d1) -> (d1, d0)>

module {
  func.func private @print_C(i32, i32, i32, i32)

  func.func @generic_transpose_i32_4x4(%A: memref<4x4xi32>,
                                       %C: memref<4x4xi32>) {
    linalg.generic {
      indexing_maps = [#transpose_in, #identity_2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : memref<4x4xi32>)
      outs(%C : memref<4x4xi32>) {
    ^bb0(%a: i32, %c: i32):
      linalg.yield %a : i32
    }
    return
  }

  func.func @main() -> i32 {
    %A = memref.alloc() : memref<4x4xi32>
    %C = memref.alloc() : memref<4x4xi32>

    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %c11_i32 = arith.constant 11 : i32
    %c12_i32 = arith.constant 12 : i32
    %c13_i32 = arith.constant 13 : i32
    %c14_i32 = arith.constant 14 : i32
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32

    linalg.fill ins(%c0_i32 : i32) outs(%C : memref<4x4xi32>)

    memref.store %c1_i32, %A[%i0, %i0] : memref<4x4xi32>
    memref.store %c2_i32, %A[%i0, %i1] : memref<4x4xi32>
    memref.store %c3_i32, %A[%i0, %i2] : memref<4x4xi32>
    memref.store %c4_i32, %A[%i0, %i3] : memref<4x4xi32>
    memref.store %c5_i32, %A[%i1, %i0] : memref<4x4xi32>
    memref.store %c6_i32, %A[%i1, %i1] : memref<4x4xi32>
    memref.store %c7_i32, %A[%i1, %i2] : memref<4x4xi32>
    memref.store %c8_i32, %A[%i1, %i3] : memref<4x4xi32>
    memref.store %c9_i32, %A[%i2, %i0] : memref<4x4xi32>
    memref.store %c10_i32, %A[%i2, %i1] : memref<4x4xi32>
    memref.store %c11_i32, %A[%i2, %i2] : memref<4x4xi32>
    memref.store %c12_i32, %A[%i2, %i3] : memref<4x4xi32>
    memref.store %c13_i32, %A[%i3, %i0] : memref<4x4xi32>
    memref.store %c14_i32, %A[%i3, %i1] : memref<4x4xi32>
    memref.store %c15_i32, %A[%i3, %i2] : memref<4x4xi32>
    memref.store %c16_i32, %A[%i3, %i3] : memref<4x4xi32>

    call @generic_transpose_i32_4x4(%A, %C)
      : (memref<4x4xi32>, memref<4x4xi32>) -> ()

    %c00 = memref.load %C[%i0, %i0] : memref<4x4xi32>
    %c01 = memref.load %C[%i0, %i1] : memref<4x4xi32>
    %c02 = memref.load %C[%i0, %i2] : memref<4x4xi32>
    %c03 = memref.load %C[%i0, %i3] : memref<4x4xi32>
    call @print_C(%c00, %c01, %c02, %c03) : (i32, i32, i32, i32) -> ()

    %c10 = memref.load %C[%i1, %i0] : memref<4x4xi32>
    %c11 = memref.load %C[%i1, %i1] : memref<4x4xi32>
    %c12 = memref.load %C[%i1, %i2] : memref<4x4xi32>
    %c13 = memref.load %C[%i1, %i3] : memref<4x4xi32>
    call @print_C(%c10, %c11, %c12, %c13) : (i32, i32, i32, i32) -> ()

    %c20 = memref.load %C[%i2, %i0] : memref<4x4xi32>
    %c21 = memref.load %C[%i2, %i1] : memref<4x4xi32>
    %c22 = memref.load %C[%i2, %i2] : memref<4x4xi32>
    %c23 = memref.load %C[%i2, %i3] : memref<4x4xi32>
    call @print_C(%c20, %c21, %c22, %c23) : (i32, i32, i32, i32) -> ()

    %c30 = memref.load %C[%i3, %i0] : memref<4x4xi32>
    %c31 = memref.load %C[%i3, %i1] : memref<4x4xi32>
    %c32 = memref.load %C[%i3, %i2] : memref<4x4xi32>
    %c33 = memref.load %C[%i3, %i3] : memref<4x4xi32>
    call @print_C(%c30, %c31, %c32, %c33) : (i32, i32, i32, i32) -> ()

    memref.dealloc %A : memref<4x4xi32>
    memref.dealloc %C : memref<4x4xi32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}

// CHECK-LABEL: func.func @generic_transpose_i32_4x4
// CHECK-SAME: (%[[A:.*]]: memref<4x4xi32>, %[[C:.*]]: memref<4x4xi32>)
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     %[[SUBVIEW_A:.*]] = memref.subview %[[A]]
// CHECK:     %[[SUBVIEW_C:.*]] = memref.subview %[[C]]
// CHECK:     bosc_ame.mlce32.m 0, %[[SUBVIEW_A]]
// CHECK:     bosc_ame.mtce32.m 1, 0
// CHECK:     bosc_ame.msce32.m 1, %[[SUBVIEW_C]]
