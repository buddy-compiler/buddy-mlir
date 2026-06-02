// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4' -cse | FileCheck %s

func.func @vir_set_vl_reduction_is_sequential(%base: memref<128xi32>,
                                              %acc: memref<i32>) {
  %c128 = arith.constant 128 : index
  vir.set_vl %c128 : index {
    %vec = vir.load %base[] : memref<128xi32> -> !vir.vec<?xi32>
    %cur = memref.load %acc[] : memref<i32>
    %next = vir.reduce %vec, %cur {kind = "maxsi"} : !vir.vec<?xi32>, i32 -> i32
    memref.store %next, %acc[] : memref<i32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @vir_set_vl_reduction_is_sequential
// CHECK-NOT: affine.parallel
// CHECK: affine.for
// CHECK: vector.load
// CHECK: vector.reduction <maxsi>
// CHECK: memref.store
