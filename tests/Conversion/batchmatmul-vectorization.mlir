// RUN: buddy-opt -batchmatmul-optimize %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK: affine.for %arg3 = #map(%c0) to #map(%dim) {
// CHECK-NEXT:   affine.prefetch %arg0[%arg3, %dim_0, %dim_2], read, locality<3>, data : memref<?x?x?xf32>
// CHECK-NEXT:   affine.for %arg4 = #map(%c0) to #map(%dim_0) {
// CHECK-NEXT:     %3 = scf.for %arg5 = %c0 to %2 step %c32 iter_args(%arg6 = %c0) -> (index) {
// CHECK-NEXT:       %6 = vector.load %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:       %7 = scf.for %arg7 = %c0 to %dim_2 step %c1 iter_args(%arg8 = %6) -> (vector<32xf32>) {
// CHECK-NEXT:         %9 = memref.load %arg0[%arg3, %arg4, %arg7] : memref<?x?x?xf32>
// CHECK-NEXT:         %10 = vector.broadcast %9 : f32 to vector<32xf32>
// CHECK-NEXT:         %11 = vector.load %arg1[%arg3, %arg7, %arg5] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:         %12 = vector.fma %10, %11, %arg8 : vector<32xf32>
// CHECK-NEXT:         scf.yield %12 : vector<32xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       vector.store %7, %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:       %8 = arith.addi %arg5, %c32 : index
// CHECK-NEXT:       scf.yield %8 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     %4 = arith.subi %dim_1, %3 : index
// CHECK-NEXT:     %5 = arith.cmpi sgt, %4, %c0 : index
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       %6 = vector.create_mask %4 : vector<32xi1>
// CHECK-NEXT:       %7 = vector.maskedload %arg2[%arg3, %arg4, %3], %6, %0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK-NEXT:       %8 = scf.for %arg5 = %c0 to %dim_2 step %c1 iter_args(%arg6 = %7) -> (vector<32xf32>) {
// CHECK-NEXT:         %9 = memref.load %arg0[%arg3, %arg4, %arg5] : memref<?x?x?xf32>
// CHECK-NEXT:         %10 = vector.broadcast %9 : f32 to vector<32xf32>
// CHECK-NEXT:         %11 = vector.maskedload %arg1[%arg3, %arg5, %3], %6, %0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK-NEXT:         %12 = vector.fma %10, %11, %arg6 : vector<32xf32>
// CHECK-NEXT:         scf.yield %12 : vector<32xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       vector.maskedstore %arg2[%arg3, %arg4, %3], %6, %8 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.batch_matmul 
    ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) 
    outs(%arg2 : memref<?x?x?xf32>)
  return
}
