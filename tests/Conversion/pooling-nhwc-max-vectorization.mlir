// RUN: buddy-opt -pooling-nhwc-max-vectorization %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK: affine.for %arg3 = #map(%c0) to #map(%dim_3) {
// CHECK-NEXT:   affine.for %arg4 = #map(%c0) to #map(%dim_4) {
// CHECK-NEXT:     affine.for %arg5 = #map(%c0) to #map(%dim_5) {
// CHECK-NEXT:       %3 = arith.muli %arg4, %c2 : index
// CHECK-NEXT:       %4 = arith.muli %arg5, %c2_0 : index
// CHECK-NEXT:       %5 = scf.for %arg6 = %c0 to %2 step %c32 iter_args(%arg7 = %c0) -> (index) {
// CHECK-NEXT:         %8 = vector.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:         %9 = affine.for %arg8 = #map(%c0) to #map(%dim) iter_args(%arg9 = %8) -> (vector<32xf32>) {
// CHECK-NEXT:           %11 = arith.addi %3, %arg8 : index
// CHECK-NEXT:           %12 = affine.for %arg10 = #map(%c0) to #map(%dim_2) iter_args(%arg11 = %arg9) -> (vector<32xf32>) {
// CHECK-NEXT:             %13 = arith.addi %4, %arg10 : index
// CHECK-NEXT:             %14 = vector.load %arg0[%arg3, %11, %13, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:             %15 = arith.maximumf %14, %arg11 : vector<32xf32>
// CHECK-NEXT:             affine.yield %15 : vector<32xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           affine.yield %12 : vector<32xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         vector.store %9, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
// CHECK-NEXT:         %10 = arith.addi %arg7, %c32 : index
// CHECK-NEXT:         scf.yield %10 : index
// CHECK-NEXT:       }
// CHECK-NEXT:       %6 = arith.subi %dim_6, %5 : index
// CHECK-NEXT:       %7 = arith.cmpi sgt, %6, %c0 : index
// CHECK-NEXT:       scf.if %7 {
// CHECK-NEXT:         %8 = vector.create_mask %6 : vector<32xi1>
// CHECK-NEXT:         %9 = vector.maskedload %arg2[%arg3, %arg4, %arg5, %5], %8, %0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK-NEXT:         %10 = affine.for %arg6 = #map(%c0) to #map(%dim) iter_args(%arg7 = %9) -> (vector<32xf32>) {
// CHECK-NEXT:           %11 = arith.addi %3, %arg6 : index
// CHECK-NEXT:           %12 = affine.for %arg8 = #map(%c0) to #map(%dim_2) iter_args(%arg9 = %arg7) -> (vector<32xf32>) {
// CHECK-NEXT:             %13 = arith.addi %arg8, %4 : index
// CHECK-NEXT:             %14 = vector.maskedload %arg0[%arg3, %11, %13, %5], %8, %0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK-NEXT:             %15 = arith.maximumf %14, %arg9 : vector<32xf32>
// CHECK-NEXT:             affine.yield %15 : vector<32xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           affine.yield %12 : vector<32xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         vector.maskedstore %arg2[%arg3, %arg4, %arg5, %5], %8, %10 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @pooling_nhwc_max(%a : memref<?x?x?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?x?x?xf32>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} 
    ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?xf32>) 
    outs(%c : memref<?x?x?x?xf32>)
  return
}
