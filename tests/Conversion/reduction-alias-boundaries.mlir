// RUN: buddy-opt %s --fold-reduction-transposes --fuse-reduction-producers | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @snapshot_before_write
// CHECK: linalg.transpose
// CHECK: memref.store
// CHECK: linalg.reduce
func.func @snapshot_before_write(%input: memref<4x8xi32>, %may_alias: memref<4x8xi32>, %seed: tensor<4xi32>, %value: i32) -> tensor<4xi32> {
  %source = bufferization.to_tensor %input restrict : memref<4x8xi32> to tensor<4x8xi32>
  %empty = tensor.empty() : tensor<8x4xi32>
  %snapshot = linalg.transpose ins(%source : tensor<4x8xi32>) outs(%empty : tensor<8x4xi32>) permutation = [1, 0]
  %zero = arith.constant 0 : index
  memref.store %value, %may_alias[%zero, %zero] : memref<4x8xi32>
  %result = linalg.reduce ins(%snapshot : tensor<8x4xi32>) outs(%seed : tensor<4xi32>) dimensions = [0]
    (%element: i32, %acc: i32) {
      %next = arith.maxsi %element, %acc : i32
      linalg.yield %next : i32
    }
  return %result : tensor<4xi32>
}

// CHECK-LABEL: func.func @shared_producer
// CHECK: linalg.generic
// CHECK: linalg.reduce
func.func @shared_producer(%input: tensor<4x?xi32>, %empty: tensor<4x?xi32>, %seed: tensor<4xi32>) -> (tensor<4x?xi32>, tensor<4xi32>) {
  %values = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel"]} ins(%input : tensor<4x?xi32>) outs(%empty : tensor<4x?xi32>) {
  ^bb0(%element: i32, %unused: i32):
    %square = arith.muli %element, %element : i32
    linalg.yield %square : i32
  } -> tensor<4x?xi32>
  %result = linalg.reduce ins(%values : tensor<4x?xi32>) outs(%seed : tensor<4xi32>) dimensions = [1]
    (%element: i32, %acc: i32) {
      %next = arith.addi %element, %acc : i32
      linalg.yield %next : i32
    }
  return %values, %result : tensor<4x?xi32>, tensor<4xi32>
}
