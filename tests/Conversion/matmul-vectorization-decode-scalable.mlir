// RUN: buddy-opt %s \
// RUN:     -matmul-vectorization-decode="vector-type=scalable vector-size=4" \
// RUN: | FileCheck %s

// Test case: m=1 (decode workload) with scalable vectors
// This tests that scalable vector types are properly generated

func.func @matmul_decode_f32(%A: memref<1x128xf32>,
                              %B: memref<128x64xf32>,
                              %C: memref<1x64xf32>) {
  linalg.matmul
    ins(%A, %B: memref<1x128xf32>, memref<128x64xf32>)
    outs(%C: memref<1x64xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_decode_f32
// CHECK:       scf.parallel
// CHECK:       vector.load {{.*}} : memref<1x64xf32>, vector<[4]xf32>
// CHECK:       scf.for
// CHECK:       memref.load {{.*}} : memref<1x128xf32>
// CHECK:       vector.broadcast {{.*}} : f32 to vector<[4]xf32>
// CHECK:       vector.load {{.*}} : memref<128x64xf32>, vector<[4]xf32>
// CHECK:       vector.fma {{.*}} : vector<[4]xf32>
// CHECK:       vector.store {{.*}} : memref<1x64xf32>, vector<[4]xf32>
// CHECK:       return
// CHECK-NOT:   linalg.matmul
