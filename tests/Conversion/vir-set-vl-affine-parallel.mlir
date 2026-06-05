// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4' -cse | FileCheck %s

func.func @vir_set_vl_affine_parallel(%base: memref<128xf32>) {
  %c128 = arith.constant 128 : index
  vir.set_vl %c128 : index {
    %vec = vir.load %base[] : memref<128xf32> -> !vir.vec<?xf32>
    vir.store %vec, %base[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @vir_set_vl_affine_parallel
// CHECK: affine.parallel
// CHECK: vector.load
// CHECK: vector.store
// CHECK: affine.for
// CHECK: memref.load
// CHECK: memref.store
