// RUN: buddy-opt -conv2d-nhwc-fhwc-vectorization %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK: affine.for %arg3 = #map(%c0) to #map(%dim) {
// CHECK-NEXT:   affine.for %arg4 = #map(%c0) to #map(%dim_6) {
// CHECK-NEXT:     affine.for %arg5 = #map(%c0) to #map(%dim_7) {
// CHECK-NEXT:       affine.for %arg6 = #map(%c0) to #map(%dim_3) {
// CHECK-NEXT:         %3 = arith.muli %arg4, %c1 : index
// CHECK-NEXT:         %4 = arith.muli %arg5, %c1_0 : index
// CHECK-NEXT:         %5 = memref.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
// CHECK-NEXT:         %6:2 = scf.for %arg7 = %c0 to %2 step %c8 iter_args(%arg8 = %c0, %arg9 = %5) -> (index, f32) {
// CHECK-NEXT:           %9 = affine.for %arg10 = #map(%c0) to #map(%dim_4) iter_args(%arg11 = %arg9) -> (f32) {
// CHECK-NEXT:             %11 = affine.for %arg12 = #map(%c0) to #map(%dim_5) iter_args(%arg13 = %arg11) -> (f32) {
// CHECK-NEXT:               %12 = arith.addi %3, %arg10 : index
// CHECK-NEXT:               %13 = arith.addi %4, %arg12 : index
// CHECK-NEXT:               %14 = vector.load %arg0[%arg3, %12, %13, %arg7] : memref<?x?x?x?xf32>, vector<8xf32>
// CHECK-NEXT:               %15 = vector.load %arg1[%arg6, %arg10, %arg12, %arg7] : memref<?x?x?x?xf32>, vector<8xf32>
// CHECK-NEXT:               %16 = arith.mulf %14, %15 : vector<8xf32>
// CHECK-NEXT:               %17 = vector.reduction <add>, %16 : vector<8xf32> into f32
// CHECK-NEXT:               %18 = arith.addf %17, %arg13 : f32
// CHECK-NEXT:               affine.yield %18 : f32
// CHECK-NEXT:             }
// CHECK-NEXT:             affine.yield %11 : f32
// CHECK-NEXT:           }
// CHECK-NEXT:           %10 = arith.addi %arg8, %c8 : index
// CHECK-NEXT:           scf.yield %10, %9 : index, f32
// CHECK-NEXT:         }
// CHECK-NEXT:         %7 = arith.subi %dim_2, %6#0 : index
// CHECK-NEXT:         %8 = arith.cmpi sgt, %7, %c0 : index
// CHECK-NEXT:         scf.if %8 {
// CHECK-NEXT:           %9 = vector.create_mask %7 : vector<8xi1>
// CHECK-NEXT:           %10 = affine.for %arg7 = #map(%c0) to #map(%dim_4) iter_args(%arg8 = %6#1) -> (f32) {
// CHECK-NEXT:             %11 = affine.for %arg9 = #map(%c0) to #map(%dim_5) iter_args(%arg10 = %arg8) -> (f32) {
// CHECK-NEXT:               %12 = arith.addi %3, %arg7 : index
// CHECK-NEXT:               %13 = arith.addi %4, %arg9 : index
// CHECK-NEXT:               %14 = vector.maskedload %arg0[%arg3, %12, %13, %6#0], %9, %0 : memref<?x?x?x?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK-NEXT:               %15 = vector.maskedload %arg1[%arg6, %arg7, %arg9, %6#0], %9, %0 : memref<?x?x?x?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK-NEXT:               %16 = arith.mulf %14, %15 : vector<8xf32>
// CHECK-NEXT:               %17 = vector.reduction <add>, %16 : vector<8xf32> into f32
// CHECK-NEXT:               %18 = arith.addf %17, %arg10 : f32
// CHECK-NEXT:               affine.yield %18 : f32
// CHECK-NEXT:             }
// CHECK-NEXT:             affine.yield %11 : f32
// CHECK-NEXT:           }
// CHECK-NEXT:           memref.store %10, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
// CHECK-NEXT:         } else {
// CHECK-NEXT:           memref.store %6#1, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
// CHECK-NEXT:         }
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
