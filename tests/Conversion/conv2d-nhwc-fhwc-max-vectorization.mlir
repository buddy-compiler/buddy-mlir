// RUN: buddy-opt -conv2d-nhwc-fhwc-vectorization %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK: affine.for %arg3 = #map(%c0) to #map(%dim) {
// CHECK-NEXT:   affine.for %arg4 = #map(%c0) to #map(%dim_4) {
// CHECK-NEXT:     affine.for %arg5 = #map(%c0) to #map(%dim_5) {
// CHECK-NEXT:       affine.for %arg6 = #map(%c0) to #map(%dim_1) {
// CHECK-NEXT:         %3:2 = scf.for %arg7 = %c0 to %2 step %c8 iter_args(%arg8 = %c0, %arg9 = %cst) -> (index, f32) {
// CHECK-NEXT:           %5 = affine.for %arg10 = #map(%c0) to #map(%dim_2) iter_args(%arg11 = %arg9) -> (f32) {
// CHECK-NEXT:             %7 = affine.for %arg12 = #map(%c0) to #map(%dim_3) iter_args(%arg13 = %arg11) -> (f32) {
// CHECK-NEXT:               %8 = arith.addi %arg4, %arg10 : index
// CHECK-NEXT:               %9 = arith.addi %arg5, %arg12 : index
// CHECK-NEXT:               %10 = vector.load %arg0[%arg3, %8, %9, %arg7] : memref<?x?x?x?xf32>, vector<8xf32>
// CHECK-NEXT:               %11 = vector.load %arg1[%arg6, %arg10, %arg12, %arg7] : memref<?x?x?x?xf32>, vector<8xf32>
// CHECK-NEXT:               %12 = arith.mulf %10, %11 : vector<8xf32>
// CHECK-NEXT:               %13 = vector.reduction <add>, %12 : vector<8xf32> into f32
// CHECK-NEXT:               %14 = arith.addf %13, %arg13 : f32
// CHECK-NEXT:               affine.yield %14 : f32
// CHECK-NEXT:             }
// CHECK-NEXT:             affine.yield %7 : f32
// CHECK-NEXT:           }
// CHECK-NEXT:           %6 = arith.addi %arg7, %c8 : index
// CHECK-NEXT:           scf.yield %6, %5 : index, f32
// CHECK-NEXT:         }
// CHECK-NEXT:         %4 = scf.for %arg7 = %c0 to %dim_0 step %c8 iter_args(%arg8 = %3#1) -> (f32) {
// CHECK-NEXT:           %5 = arith.subi %dim_0, %3#0 : index
// CHECK-NEXT:           %6 = vector.create_mask %5 : vector<8xi1>
// CHECK-NEXT:           %7 = affine.for %arg9 = #map(%c0) to #map(%dim_2) iter_args(%arg10 = %arg8) -> (f32) {
// CHECK-NEXT:             %8 = affine.for %arg11 = #map(%c0) to #map(%dim_3) iter_args(%arg12 = %arg10) -> (f32) {
// CHECK-NEXT:               %9 = arith.addi %arg4, %arg9 : index
// CHECK-NEXT:               %10 = arith.addi %arg5, %arg11 : index
// CHECK-NEXT:               %11 = vector.maskedload %arg0[%arg3, %9, %10, %3#0], %6, %0 : memref<?x?x?x?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK-NEXT:               %12 = vector.maskedload %arg1[%arg6, %arg9, %arg11, %3#0], %6, %0 : memref<?x?x?x?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK-NEXT:               %13 = arith.mulf %11, %12 : vector<8xf32>
// CHECK-NEXT:               %14 = vector.reduction <add>, %13 : vector<8xf32> into f32
// CHECK-NEXT:               %15 = arith.addf %14, %arg12 : f32
// CHECK-NEXT:               affine.yield %15 : f32
// CHECK-NEXT:             }
// CHECK-NEXT:             affine.yield %8 : f32
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %7 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         memref.store %4, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @conv_2d_nhwc_fhwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
          ins (%arg0, %arg1: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
          outs (%arg2: memref<?x?x?x?xf32>)
  return
}
