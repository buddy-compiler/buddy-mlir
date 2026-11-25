// RUN: buddy-opt %s --forward-masked-loads | FileCheck %s

// CHECK-LABEL: func.func @multiple_consumers
// CHECK-NOT: memref.alloc
// CHECK: vector.create_mask
// CHECK: vector.maskedload
// CHECK: scf.if
// CHECK: memref.load
func.func @multiple_consumers(%input: memref<64xi32>, %count: index, %other: i32, %pos: index) -> (vector<4xi32>, i32) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %length = arith.constant 64 : index
  %tile = memref.alloc() : memref<64xi32>
  scf.for %idx = %zero to %length step %one {
    memref.store %other, %tile[%idx] : memref<64xi32>
  }
  %src = memref.subview %input[0] [%count] [1] : memref<64xi32> to memref<?xi32, strided<[1]>>
  %dst = memref.subview %tile[0] [%count] [1] : memref<64xi32> to memref<?xi32, strided<[1]>>
  %quotient = arith.divui %count, %four : index
  %main = arith.muli %quotient, %four : index
  scf.for %idx = %zero to %main step %four {
    %values = vector.load %src[%idx] : memref<?xi32, strided<[1]>>, vector<4xi32>
    vector.store %values, %dst[%idx] : memref<?xi32, strided<[1]>>, vector<4xi32>
  }
  scf.for %idx = %main to %count step %one {
    %value = memref.load %src[%idx] : memref<?xi32, strided<[1]>>
    memref.store %value, %dst[%idx] : memref<?xi32, strided<[1]>>
  }
  %result = vector.load %tile[%pos] : memref<64xi32>, vector<4xi32>
  %scalar = memref.load %tile[%pos] : memref<64xi32>
  memref.dealloc %tile : memref<64xi32>
  return %result, %scalar : vector<4xi32>, i32
}

// CHECK-LABEL: func.func @alias_write
// CHECK: memref.alloc
// CHECK-NOT: vector.maskedload
// CHECK: return
func.func @alias_write(%input: memref<64xi32>, %may_alias: memref<64xi32>, %count: index, %overwrite: i32) -> vector<4xi32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %tile = memref.alloc() : memref<64xi32>
  %src = memref.subview %input[0] [%count] [1] : memref<64xi32> to memref<?xi32, strided<[1]>>
  %dst = memref.subview %tile[0] [%count] [1] : memref<64xi32> to memref<?xi32, strided<[1]>>
  %quotient = arith.divui %count, %four : index
  %main = arith.muli %quotient, %four : index
  scf.for %idx = %zero to %main step %four {
    %values = vector.load %src[%idx] : memref<?xi32, strided<[1]>>, vector<4xi32>
    vector.store %values, %dst[%idx] : memref<?xi32, strided<[1]>>, vector<4xi32>
  }
  scf.for %idx = %main to %count step %one {
    %value = memref.load %src[%idx] : memref<?xi32, strided<[1]>>
    memref.store %value, %dst[%idx] : memref<?xi32, strided<[1]>>
  }
  memref.store %overwrite, %may_alias[%zero] : memref<64xi32>
  %result = vector.load %tile[%zero] : memref<64xi32>, vector<4xi32>
  return %result : vector<4xi32>
}
