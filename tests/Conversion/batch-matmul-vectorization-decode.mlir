// RUN: buddy-opt %s -batch-matmul-vectorization-decode | FileCheck %s

// Build a decode-like batch matmul: M=1 so loops specialize to [batch, 0, *].
// Expect vector.load/store on C, broadcast of A[b,0,k], and fma with B[b,k,n].
func.func @batch_decode(%A: memref<12x1x1024xf32>,
                        %B: memref<12x1024x128xf32>,
                        %C: memref<12x1x128xf32>) {
  // linalg on buffers form
  linalg.batch_matmul
    ins(%A, %B : memref<12x1x1024xf32>, memref<12x1024x128xf32>)
    outs(%C : memref<12x1x128xf32>)
  return
}

// CHECK-LABEL: func.func @batch_decode
// CHECK:       scf.parallel
// CHECK:       scf.for
// CHECK:       vector.load {{.*}} : memref<12x1x128xf32>, vector<
// CHECK:       scf.for
// CHECK:       memref.load {{.*}} : memref<12x1x1024xf32>
// CHECK:       vector.broadcast
// CHECK:       vector.load {{.*}} : memref<12x1024x128xf32>, vector<
// CHECK:       vector.fma
// CHECK:       vector.store {{.*}} : memref<12x1x128xf32>, vector<
// CHECK:       return
// CHECK-NOT:   linalg.batch_matmul
