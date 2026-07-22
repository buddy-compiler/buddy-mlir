// RUN: buddy-opt %s -lower-linalg-to-boscame | FileCheck %s
//
// This file tests lowering a linalg.generic elementwise add pattern to BOSCAME.

#identity_2d = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func private @print_C(i32, i32, i32, i32)

  func.func @generic_elementwise_add_i32_4x4(%A: memref<4x4xi32>,
                                             %B: memref<4x4xi32>,
                                             %C: memref<4x4xi32>) {
    linalg.generic {
      indexing_maps = [#identity_2d, #identity_2d, #identity_2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%A, %B : memref<4x4xi32>, memref<4x4xi32>)
      outs(%C : memref<4x4xi32>) {
    ^bb0(%a: i32, %b: i32, %c: i32):
      %add = arith.addi %a, %b : i32
      linalg.yield %add : i32
    }
    return
  }

  func.func @main() -> i32 {
    %A = memref.alloc() : memref<4x4xi32>
    %B = memref.alloc() : memref<4x4xi32>
    %C = memref.alloc() : memref<4x4xi32>

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32

    linalg.fill ins(%c1_i32 : i32) outs(%A : memref<4x4xi32>)
    linalg.fill ins(%c2_i32 : i32) outs(%B : memref<4x4xi32>)
    linalg.fill ins(%c0_i32 : i32) outs(%C : memref<4x4xi32>)

    call @generic_elementwise_add_i32_4x4(%A, %B, %C)
      : (memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>) -> ()

    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

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
    memref.dealloc %B : memref<4x4xi32>
    memref.dealloc %C : memref<4x4xi32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}

// CHECK-LABEL: func.func @generic_elementwise_add_i32_4x4
// CHECK-SAME: (%[[A:.*]]: memref<4x4xi32>, %[[B:.*]]: memref<4x4xi32>, %[[C:.*]]: memref<4x4xi32>)
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     %[[SUBVIEW_A:.*]] = memref.subview %[[A]]
// CHECK:     %[[SUBVIEW_B:.*]] = memref.subview %[[B]]
// CHECK:     %[[SUBVIEW_C:.*]] = memref.subview %[[C]]
// CHECK:     bosc_ame.mlce32.m 0, %[[SUBVIEW_A]]
// CHECK:     bosc_ame.mlce32.m 1, %[[SUBVIEW_B]]
// CHECK:     bosc_ame.madd.w.mm 2, 0, 1
// CHECK:     bosc_ame.msce32.m 2, %[[SUBVIEW_C]]
