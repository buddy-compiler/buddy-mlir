// RUN: buddy-opt %s --vectorize-masked-stores | FileCheck %s

// CHECK-LABEL: func.func @borrowed_values
// CHECK-NOT: vector.maskedstore
// CHECK: memref.store
func.func @borrowed_values(%destination: memref<16xi32>, %mask: tensor<16xi1>) {
  %values = bufferization.to_tensor %destination restrict writable : memref<16xi32> to tensor<16xi32>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %upper = arith.constant 16 : index
  scf.for %pos = %zero to %upper step %one {
    %value = tensor.extract %values[%pos] : tensor<16xi32>
    %pred = tensor.extract %mask[%pos] : tensor<16xi1>
    scf.if %pred {
      memref.store %value, %destination[%pos] : memref<16xi32>
    }
  }
  return
}
