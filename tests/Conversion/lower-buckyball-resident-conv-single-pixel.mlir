// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s

// CHECK-LABEL: func.func @single_pixel_k16(
// CHECK: %[[CFG_SINGLE_PIXEL_K16:[a-zA-Z0-9_]+]] = arith.constant 268500993 : i64
// CHECK: %[[C16_CAST:[a-zA-Z0-9_]+]] = memref.cast %arg0 : memref<1x1x1x16xi8> to memref<1x1x?x?xi8
// CHECK: %[[C16_DMA:[a-zA-Z0-9_]+]] = buckyball.bank_mvin_2d %[[C16_CAST]] %{{[a-zA-Z0-9_]+}} %c1_i64 %c16_i64 %c1_i64 %c0_i64 %c1_i64 %c16_i64
// CHECK: buckyball.bank_maxpool %[[C16_DMA]]
// CHECK: buckyball.bank_smatmul {{.*}}%[[CFG_SINGLE_PIXEL_K16]]
func.func @single_pixel_k16(%input: memref<1x1x1x16xi8>, %weight: memref<1x16x16x16xi8>, %bias: memref<16xi32>, %scale: memref<16xf32>, %lut: memref<1xi8>, %output: memref<1x1x1x16xi8>) {
  buckyball.mega_kernel %input %output : memref<1x1x1x16xi8> memref<1x1x1x16xi8> {
    buckyball.mega_conv2d %input %weight %bias %scale %lut %output {activation = 0 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x1x1x16xi8> memref<1x16x16x16xi8> memref<16xi32> memref<16xf32> memref<1xi8> memref<1x1x1x16xi8>
  }
  return
}

// CHECK-LABEL: func.func @single_pixel_k32(
// CHECK: %[[CFG_SINGLE_PIXEL_K32:[a-zA-Z0-9_]+]] = arith.constant 536936464 : i64
// CHECK: buckyball.bank_smatmul {{.*}}%[[CFG_SINGLE_PIXEL_K32]]
func.func @single_pixel_k32(%input: memref<1x5x5x16xi8>, %weight: memref<1x16x32x16xi8>, %bias: memref<16xi32>, %scale: memref<16xf32>, %lut: memref<1xi8>, %output: memref<1x1x1x16xi8>) {
  buckyball.mega_kernel %input %output : memref<1x5x5x16xi8> memref<1x1x1x16xi8> {
    buckyball.mega_conv2d %input %weight %bias %scale %lut %output {activation = 0 : i64, kernel = 5 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x5x5x16xi8> memref<1x16x32x16xi8> memref<16xi32> memref<16xf32> memref<1xi8> memref<1x1x1x16xi8>
  }
  return
}

// CHECK-LABEL: func.func @multiple_pixels_k16(
// CHECK: %[[CFG_MULTIPLE_PIXELS_K16:[a-zA-Z0-9_]+]] = arith.constant 268501008 : i64
// CHECK: buckyball.bank_smatmul {{.*}}%[[CFG_MULTIPLE_PIXELS_K16]]
func.func @multiple_pixels_k16(%input: memref<1x2x2x16xi8>, %weight: memref<1x16x16x16xi8>, %bias: memref<16xi32>, %scale: memref<16xf32>, %lut: memref<1xi8>, %output: memref<1x2x2x16xi8>) {
  buckyball.mega_kernel %input %output : memref<1x2x2x16xi8> memref<1x2x2x16xi8> {
    buckyball.mega_conv2d %input %weight %bias %scale %lut %output {activation = 0 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x2x2x16xi8> memref<1x16x16x16xi8> memref<16xi32> memref<16xf32> memref<1xi8> memref<1x2x2x16xi8>
  }
  return
}

// CHECK-LABEL: func.func @external_cin3_scalar(
// CHECK-NOT: buckyball.bank_mvin_2d
// CHECK: buckyball.bank_smatmul_bias
// CHECK: %[[C3_PACK:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64x16xi8>
// CHECK: linalg.fill ins(%c0_i8 : i8) outs(%[[C3_PACK]] : memref<64x16xi8>)
// CHECK: scf.for %[[C3_LANE:[a-zA-Z0-9_]+]] = %c0 to %c16 step %c1
// CHECK: %[[C3_VALID:[a-zA-Z0-9_]+]] = arith.cmpi slt, %[[C3_LANE]], %c3 : index
// CHECK-NEXT: scf.if %[[C3_VALID]]
// CHECK-NEXT: %[[C3_BYTE:[a-zA-Z0-9_]+]] = memref.load %arg0[%c0, %c0, %c0, %[[C3_LANE]]]
// CHECK-NEXT: memref.store %[[C3_BYTE]], %[[C3_PACK]][%c0, %[[C3_LANE]]]
// CHECK: %[[C3_LOADED:[a-zA-Z0-9_]+]] = buckyball.bank_mvin %[[C3_PACK]]
// CHECK: buckyball.bank_im2col %[[C3_LOADED]]
// CHECK-NOT: buckyball.bank_mvin_2d
// CHECK: return
func.func @external_cin3_scalar(%input: memref<1x1x1x3xi8>, %weight: memref<1x3x16x16xi8>, %bias: memref<16xi32>, %scale: memref<16xf32>, %lut: memref<1xi8>, %output: memref<1x1x1x16xi8>) {
  buckyball.mega_kernel %input %output : memref<1x1x1x3xi8> memref<1x1x1x16xi8> {
    buckyball.mega_conv2d %input %weight %bias %scale %lut %output {activation = 0 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x1x1x3xi8> memref<1x3x16x16xi8> memref<16xi32> memref<16xf32> memref<1xi8> memref<1x1x1x16xi8>
  }
  return
}
