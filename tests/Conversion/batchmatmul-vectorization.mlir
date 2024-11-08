// RUN: buddy-opt -batchmatmul-optimize %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK: affine.for %arg3 = #map(%c0) to #map(%dim) {
// CHECK-NEXT:   affine.prefetch %arg0[%arg3, %dim_0, %dim_2], read, locality<3>, data : memref<?x?x?xf32>
// CHECK-NEXT:   affine.for %arg4 = #map(%c0) to #map(%dim_0) {
// CHECK-NEXT:     %3 = scf.for %arg5 = %c0 to %2 step %c32 iter_args(%arg6 = %c0) -> (index) {
// CHECK-NEXT:       %8 = vector.load %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:       %9 = scf.for %arg7 = %c0 to %dim_2 step %c1 iter_args(%arg8 = %8) -> (vector<32xf32>) {
// CHECK-NEXT:         %11 = memref.load %arg0[%arg3, %arg4, %arg7] : memref<?x?x?xf32>
// CHECK-NEXT:         %12 = vector.broadcast %11 : f32 to vector<32xf32>
// CHECK-NEXT:         %13 = vector.load %arg1[%arg3, %arg7, %arg5] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:         %14 = vector.fma %12, %13, %arg8 : vector<32xf32>
// CHECK-NEXT:         scf.yield %14 : vector<32xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       vector.store %9, %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:       %10 = arith.addi %arg5, %c32 : index
// CHECK-NEXT:       scf.yield %10 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     %4 = arith.subi %dim_1, %3 : index
// CHECK-NEXT:     %5 = vector.create_mask %4 : vector<32xi1>
// CHECK-NEXT:     %6 = vector.maskedload %arg2[%arg3, %arg4, %3], %5, %0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK-NEXT:     %7 = scf.for %arg5 = %c0 to %dim_2 step %c1 iter_args(%arg6 = %6) -> (vector<32xf32>) {
// CHECK-NEXT:       %8 = memref.load %arg0[%arg3, %arg4, %arg5] : memref<?x?x?xf32>
// CHECK-NEXT:       %9 = vector.broadcast %8 : f32 to vector<32xf32>
// CHECK-NEXT:       %10 = vector.maskedload %arg1[%arg3, %arg5, %3], %5, %0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK-NEXT:       %11 = vector.fma %9, %10, %arg6 : vector<32xf32>
// CHECK-NEXT:       scf.yield %11 : vector<32xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     vector.maskedstore %arg2[%arg3, %arg4, %3], %5, %7 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.batch_matmul 
    ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) 
    outs(%arg2 : memref<?x?x?xf32>)
  return
}
