// RUN: buddy-opt %s -split-input-file \
// RUN:   -pass-pipeline="builtin.module(func.func(convert-strided-memref-copy-to-linalg))" \
// RUN: | FileCheck %s

// CHECK-LABEL: func @strided_copy
//   CHECK-NOT: memref.copy
//       CHECK: linalg.generic
//  CHECK-SAME: iterator_types = ["parallel", "parallel"]
//       CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
//       CHECK: linalg.yield %[[IN]] : f32
func.func @strided_copy(%arg0: memref<4x8xf32, strided<[?, ?], offset: ?>>,
                        %arg1: memref<4x8xf32>) {
  memref.copy %arg0, %arg1
    : memref<4x8xf32, strided<[?, ?], offset: ?>> to memref<4x8xf32>
  return
}

// -----

// CHECK-LABEL: func @identity_copy
//       CHECK: memref.copy
//   CHECK-NOT: linalg.generic
func.func @identity_copy(%arg0: memref<4x8xf32>, %arg1: memref<4x8xf32>) {
  memref.copy %arg0, %arg1 : memref<4x8xf32> to memref<4x8xf32>
  return
}

// -----

// CHECK-LABEL: func @subview_copy
//       CHECK: %[[SRC:.*]] = memref.subview
//       CHECK: %[[DST:.*]] = memref.subview
//   CHECK-NOT: memref.copy
//       CHECK: linalg.generic
//  CHECK-SAME: ins(%[[SRC]]
//  CHECK-SAME: outs(%[[DST]]
//       CHECK: linalg.yield
func.func @subview_copy(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %src = memref.subview %arg0[%c0, %c0] [4, 4] [1, 1]
    : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1], offset: ?>>
  %dst = memref.subview %arg1[%c0, %c0] [4, 4] [1, 1]
    : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1], offset: ?>>
  memref.copy %src, %dst
    : memref<4x4xf32, strided<[8, 1], offset: ?>>
      to memref<4x4xf32, strided<[8, 1], offset: ?>>
  return
}
