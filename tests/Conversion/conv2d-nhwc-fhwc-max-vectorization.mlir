// RUN: buddy-opt -conv2d-nhwc-fhwc-vectorization %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK: affine.for %arg3 = #map(%c0) to #map(%dim) {
// CHECK-NEXT:         affine.for %arg4 = #map(%c0) to #map(%dim_8) {
// CHECK-NEXT:           affine.for %arg5 = #map(%c0) to #map(%dim_9) {
// CHECK-NEXT:             affine.for %arg6 = #map(%c0) to #map(%dim_5) {
// CHECK-NEXT:               %3 = memref.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
// CHECK-NEXT:               %4:2 = scf.for %arg7 = %c0 to %2 step %c8 iter_args(%arg8 = %c0, %arg9 = %3) -> (index, f32) {
// CHECK-NEXT:                 %8 = affine.for %arg10 = #map(%c0) to #map(%dim_6) iter_args(%arg11 = %arg9) -> (f32) {
// CHECK-NEXT:                   %10 = arith.addi %arg4, %arg10 : index
// CHECK-NEXT:                   %11 = affine.for %arg12 = #map(%c0) to #map(%dim_7) iter_args(%arg13 = %arg11) -> (f32) {
// CHECK-NEXT:                     %12 = arith.addi %arg5, %arg12 : index
// CHECK-NEXT:                     %13 = vector.load %arg0[%arg3, %10, %12, %arg7] : memref<?x?x?x?xf32>, vector<8xf32>
// CHECK-NEXT:                     %14 = vector.load %arg1[%arg6, %arg10, %arg12, %arg7] : memref<?x?x?x?xf32>, vector<8xf32>
// CHECK-NEXT:                     %15 = arith.mulf %13, %14 : vector<8xf32>
// CHECK-NEXT:                     %16 = vector.reduction <add>, %15, %arg13 fastmath<reassoc> : vector<8xf32> into f32
// CHECK-NEXT:                     affine.yield %16 : f32
// CHECK-NEXT:                   }
// CHECK-NEXT:                   affine.yield %11 : f32
// CHECK-NEXT:                 }
// CHECK-NEXT:                 %9 = arith.addi %arg7, %c8 : index
// CHECK-NEXT:                 scf.yield %9, %8 : index, f32
// CHECK-NEXT:               }

func.func @conv_2d_nhwc_fhwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
          ins (%arg0, %arg1: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
          outs (%arg2: memref<?x?x?x?xf32>)
  return
}
