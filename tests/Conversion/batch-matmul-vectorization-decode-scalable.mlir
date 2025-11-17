// RUN: buddy-opt %s \
// RUN:     -batch-matmul-vectorization-decode="vector-type=scalable vector-size=4" \
// RUN: | FileCheck %s

// Test case: batch matmul with m=1 (decode workload) with scalable vectors
// This tests that scalable vector types are properly generated

func.func @batch_matmul_decode_f32(%A: memref<8x1x128xf32>,
                                    %B: memref<8x128x64xf32>,
                                    %C: memref<8x1x64xf32>) {
  linalg.batch_matmul
    ins(%A, %B: memref<8x1x128xf32>, memref<8x128x64xf32>)
    outs(%C: memref<8x1x64xf32>)
  return
}

// CHECK-LABEL: func.func @batch_matmul_decode_f32
// CHECK:       scf.parallel
// CHECK:       scf.for
// CHECK:       vector.load {{.*}} : memref<8x1x64xf32>, vector<[4]xf32>
// CHECK:       scf.for
// CHECK:       memref.load {{.*}} : memref<8x1x128xf32>
// CHECK:       vector.broadcast {{.*}} : f32 to vector<[4]xf32>
// CHECK:       vector.load {{.*}} : memref<8x128x64xf32>, vector<[4]xf32>
// CHECK:       vector.fma {{.*}} : vector<[4]xf32>
// CHECK:       vector.store {{.*}} : memref<8x1x64xf32>, vector<[4]xf32>
// CHECK:       return
// CHECK-NOT:   linalg.batch_matmul
