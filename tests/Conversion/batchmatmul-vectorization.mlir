// RUN: buddy-opt -batchmatmul-optimize %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0 mod 32)>
// CHECK: #map1 = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK: affine.parallel (%arg3) = (0) to (%dim) {
// CHECK-NEXT:   affine.prefetch %arg0[%arg3, %dim_0, %dim_2], read, locality<3>, data : memref<?x?x?xf32>
// CHECK-NEXT:   %5 = affine.for %arg4 = #map1(%c0) to #map1(%2) step 32 iter_args(%arg5 = %c0) -> (index) {
// CHECK-NEXT:     affine.for %arg6 = #map1(%c0) to #map1(%dim_2) {
// CHECK-NEXT:       %7 = affine.vector_load %arg1[%arg3, %arg6, %arg4] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:       affine.for %arg7 = #map1(%c0) to #map1(%dim_0) {
// CHECK-NEXT:         %8 = memref.load %arg0[%arg3, %arg7, %arg6] : memref<?x?x?xf32>
// CHECK-NEXT:         %9 = vector.broadcast %8 : f32 to vector<32xf32>
// CHECK-NEXT:         %10 = affine.vector_load %arg2[%arg3, %arg7, %arg4] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:         %11 = vector.fma %9, %7, %10 : vector<32xf32>
// CHECK-NEXT:         affine.vector_store %11, %arg2[%arg3, %arg7, %arg4] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     %6 = arith.addi %arg4, %c32 : index
// CHECK-NEXT:     affine.yield %6 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %arg4 = #map1(%c0) to #map1(%dim_2) {
// CHECK-NEXT:     %6 = vector.maskedload %arg1[%arg3, %arg4, %5], %4, %0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK-NEXT:     affine.for %arg5 = #map1(%c0) to #map1(%dim_0) {
// CHECK-NEXT:       %7 = memref.load %arg0[%arg3, %arg5, %arg4] : memref<?x?x?xf32>
// CHECK-NEXT:       %8 = vector.broadcast %7 : f32 to vector<32xf32>
// CHECK-NEXT:       %9 = vector.maskedload %arg2[%arg3, %arg5, %5], %4, %0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK-NEXT:       %10 = vector.fma %8, %6, %9 : vector<32xf32>
// CHECK-NEXT:       vector.maskedstore %arg2[%arg3, %arg5, %5], %4, %10 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.batch_matmul 
    ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) 
    outs(%arg2 : memref<?x?x?xf32>)
  return
}
