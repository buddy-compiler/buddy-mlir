// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s
// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s --check-prefix=LOAD

// CHECK-LABEL: func.func @depthwise_external_tail(
// CHECK: %[[WEIGHTS:[a-zA-Z0-9_]+]] = memref.alloc() : memref<2x16x16x16xi8>
// CHECK: linalg.fill ins(%c0_i8 : i8) outs(%[[WEIGHTS]] : memref<2x16x16x16xi8>)
// CHECK: buckyball.bank_mvin_2d
// CHECK: buckyball.bank_im2col
// CHECK-NEXT: %[[SLICE:[a-zA-Z0-9_]+]] = memref.subview %[[WEIGHTS]]
// CHECK-NEXT: %[[PACK:[a-zA-Z0-9_]+]] = memref.collapse_shape %[[SLICE]]
// CHECK-NEXT: {{.*}}buckyball.bank_mvin %[[PACK]]
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.dealloc
// CHECK: buckyball.bank_smatmul
// CHECK: return
// LOAD-LABEL: func.func @depthwise_external_tail(
// LOAD-NOT: memref.load %arg0
// LOAD: return
func.func @depthwise_external_tail(%input: memref<1x9x9x24xi8>, %weight: memref<3x3x24x1xi8>, %bias: memref<24xi32>, %scale: memref<24xf32>, %lut: memref<1xi8>, %output: memref<1x9x9x24xi8>) {
  buckyball.mega_kernel %input %output : memref<1x9x9x24xi8> memref<1x9x9x24xi8> {
    buckyball.mega_conv2d_depthwise %input %weight %bias %scale %lut %output {activation = 0 : i64, kernel = 3 : i64, outputScale = 1.0 : f32, padHigh = 1 : i64, padLow = 1 : i64, stride = 1 : i64} : memref<1x9x9x24xi8> memref<3x3x24x1xi8> memref<24xi32> memref<24xf32> memref<1xi8> memref<1x9x9x24xi8>
  }
  return
}
