// RUN: buddy-opt %s -matmul-vectorization-decode | FileCheck %s

memref.global "private" constant @zero_out : memref<1x64xf32> = dense<0.0>

func.func @matmul_decode_zero_init(%A: memref<1x128xf32>,
                                   %B: memref<128x64xf32>) {
  %zero = memref.get_global @zero_out : memref<1x64xf32>
  %C = memref.alloc() : memref<1x64xf32>
  memref.copy %zero, %C : memref<1x64xf32> to memref<1x64xf32>
  linalg.matmul
    ins(%A, %B: memref<1x128xf32>, memref<128x64xf32>)
    outs(%C: memref<1x64xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_decode_zero_init
// CHECK-NOT: memref.copy
// CHECK: scf.parallel
// CHECK-NOT: vector.load {{.*}} : memref<1x64xf32>
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: vector.broadcast {{.*}} : f32 to vector<32xf32>
// CHECK: vector.store
// CHECK-NOT: linalg.matmul
