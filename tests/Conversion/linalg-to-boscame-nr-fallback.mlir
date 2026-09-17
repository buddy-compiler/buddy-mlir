// RUN: buddy-opt %s --lower-linalg-to-boscame=target=nr-fpga | FileCheck %s --implicit-check-not=bosc_ame.m

// Tensor form must be bufferized before AME lowering. Do not introduce memory
// effects into a tensor value operation during this pass.
func.func @tensor_input(%a: tensor<3x70xi8>, %b: tensor<70x19xi8>,
                        %c: tensor<3x19xi32>) -> tensor<3x19xi32> {
  %r = linalg.matmul ins(%a, %b : tensor<3x70xi8>, tensor<70x19xi8>)
      outs(%c : tensor<3x19xi32>) -> tensor<3x19xi32>
  return %r : tensor<3x19xi32>
}
// CHECK-LABEL: func.func @tensor_input
// CHECK: linalg.matmul
// CHECK: tensor<3x19xi32>

// This implementation owns static geometry only. A later CPU lowering must
// still see the entire dynamic matmul, rather than a partially changed op.
func.func @dynamic_shape(%a: memref<?x?xi8>, %b: memref<?x?xi8>,
                         %c: memref<?x?xi32>) {
  linalg.matmul ins(%a, %b : memref<?x?xi8>, memref<?x?xi8>)
      outs(%c : memref<?x?xi32>)
  return
}
// CHECK-LABEL: func.func @dynamic_shape
// CHECK: linalg.matmul

func.func @unsigned_input(%a: memref<3x70xi8>, %b: memref<70x19xi8>,
                          %c: memref<3x19xi32>) {
  linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
      ins(%a, %b : memref<3x70xi8>, memref<70x19xi8>)
      outs(%c : memref<3x19xi32>)
  return
}
// CHECK-LABEL: func.func @unsigned_input
// CHECK: linalg.matmul
// CHECK-SAME: cast_unsigned

func.func @noncontiguous_a(%a: memref<3x70xi8, strided<[140, 2]>>,
                           %b: memref<70x19xi8>, %c: memref<3x19xi32>) {
  linalg.matmul ins(%a, %b : memref<3x70xi8, strided<[140, 2]>>,
      memref<70x19xi8>) outs(%c : memref<3x19xi32>)
  return
}
// CHECK-LABEL: func.func @noncontiguous_a
// CHECK: linalg.matmul
