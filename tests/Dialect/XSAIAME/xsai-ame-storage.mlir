// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// CHECK-LABEL: func.func @scalar_storage
func.func @scalar_storage(%i4: memref<4x8xi4>, %u4: memref<4x8xui4>,
                          %u8: memref<4x8xui8>, %f16: memref<4x8xf16>,
                          %bf16: memref<4x8xbf16>, %f32: memref<4x8xf32>) {
  // CHECK: xsai_ame.load {{.*}} : memref<4x8xi4> -> !xsai_ame.tile<a, 4x8, i4>
  %a = xsai_ame.load %i4 : memref<4x8xi4> -> !xsai_ame.tile<a, 4x8, i4>
  // CHECK: xsai_ame.load {{.*}} : memref<4x8xui4> -> !xsai_ame.tile<a, 4x8, u4>
  %b = xsai_ame.load %u4 : memref<4x8xui4> -> !xsai_ame.tile<a, 4x8, u4>
  // CHECK: xsai_ame.load {{.*}} : memref<4x8xui8> -> !xsai_ame.tile<b, 4x8, u8>
  %c = xsai_ame.load %u8 : memref<4x8xui8> -> !xsai_ame.tile<b, 4x8, u8>
  // CHECK: xsai_ame.load {{.*}} : memref<4x8xf16> -> !xsai_ame.tile<a, 4x8, fp16>
  %d = xsai_ame.load %f16 : memref<4x8xf16> -> !xsai_ame.tile<a, 4x8, fp16>
  // CHECK: xsai_ame.load {{.*}} : memref<4x8xbf16> -> !xsai_ame.tile<b, 4x8, bf16>
  %e = xsai_ame.load %bf16 : memref<4x8xbf16> -> !xsai_ame.tile<b, 4x8, bf16>
  // CHECK: xsai_ame.load {{.*}} : memref<4x8xf32> -> !xsai_ame.tile<a, 4x8, tf32>
  %f = xsai_ame.load %f32 : memref<4x8xf32> -> !xsai_ame.tile<a, 4x8, tf32>
  // CHECK: xsai_ame.load {{.*}} : memref<4x8xf32> -> !xsai_ame.tile<c, 4x8, fp32>
  %g = xsai_ame.load %f32 : memref<4x8xf32> -> !xsai_ame.tile<c, 4x8, fp32>
  xsai_ame.store %a, %i4 : !xsai_ame.tile<a, 4x8, i4>, memref<4x8xi4>
  xsai_ame.store %b, %u4 : !xsai_ame.tile<a, 4x8, u4>, memref<4x8xui4>
  xsai_ame.store %c, %u8 : !xsai_ame.tile<b, 4x8, u8>, memref<4x8xui8>
  xsai_ame.store %d, %f16 : !xsai_ame.tile<a, 4x8, fp16>, memref<4x8xf16>
  xsai_ame.store %e, %bf16 : !xsai_ame.tile<b, 4x8, bf16>, memref<4x8xbf16>
  xsai_ame.store %f, %f32 : !xsai_ame.tile<a, 4x8, tf32>, memref<4x8xf32>
  xsai_ame.store %g, %f32 : !xsai_ame.tile<c, 4x8, fp32>, memref<4x8xf32>
  return
}

// These tokens have no one-to-one builtin storage mapping at L1. Their native
// lowering support is intentionally not implied by representability here.
// CHECK-LABEL: func.func @historical_packed_storage
func.func @historical_packed_storage(%view: memref<4x8xi8>) {
  // CHECK: xsai_ame.load {{.*}} -> !xsai_ame.tile<a, 4x8, nvfp4>
  %a = xsai_ame.load %view : memref<4x8xi8> -> !xsai_ame.tile<a, 4x8, nvfp4>
  // CHECK: xsai_ame.load {{.*}} -> !xsai_ame.tile<b, 4x8, mxfp4>
  %b = xsai_ame.load %view : memref<4x8xi8> -> !xsai_ame.tile<b, 4x8, mxfp4>
  // CHECK: xsai_ame.load {{.*}} -> !xsai_ame.tile<a, 4x8, fp2pack4>
  %c = xsai_ame.load %view : memref<4x8xi8> -> !xsai_ame.tile<a, 4x8, fp2pack4>
  // CHECK: xsai_ame.load {{.*}} -> !xsai_ame.tile<b, 4x8, fp2pack5>
  %d = xsai_ame.load %view : memref<4x8xi8> -> !xsai_ame.tile<b, 4x8, fp2pack5>
  xsai_ame.store %a, %view : !xsai_ame.tile<a, 4x8, nvfp4>, memref<4x8xi8>
  xsai_ame.store %b, %view : !xsai_ame.tile<b, 4x8, mxfp4>, memref<4x8xi8>
  xsai_ame.store %c, %view : !xsai_ame.tile<a, 4x8, fp2pack4>, memref<4x8xi8>
  xsai_ame.store %d, %view : !xsai_ame.tile<b, 4x8, fp2pack5>, memref<4x8xi8>
  return
}
