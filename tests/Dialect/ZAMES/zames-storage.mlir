// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// CHECK-LABEL: func.func @scalar_storage
func.func @scalar_storage(%i4: memref<4x8xi4>, %u4: memref<4x8xui4>,
                          %u8: memref<4x8xui8>, %f16: memref<4x8xf16>,
                          %bf16: memref<4x8xbf16>, %f32: memref<4x8xf32>) {
  // CHECK: zames.load {{.*}} : memref<4x8xi4> -> !zames.tile<a, 4x8, i4>
  %a = zames.load %i4 : memref<4x8xi4> -> !zames.tile<a, 4x8, i4>
  // CHECK: zames.load {{.*}} : memref<4x8xui4> -> !zames.tile<a, 4x8, u4>
  %b = zames.load %u4 : memref<4x8xui4> -> !zames.tile<a, 4x8, u4>
  // CHECK: zames.load {{.*}} : memref<4x8xui8> -> !zames.tile<b, 4x8, u8>
  %c = zames.load %u8 : memref<4x8xui8> -> !zames.tile<b, 4x8, u8>
  // CHECK: zames.load {{.*}} : memref<4x8xf16> -> !zames.tile<a, 4x8, fp16>
  %d = zames.load %f16 : memref<4x8xf16> -> !zames.tile<a, 4x8, fp16>
  // CHECK: zames.load {{.*}} : memref<4x8xbf16> -> !zames.tile<b, 4x8, bf16>
  %e = zames.load %bf16 : memref<4x8xbf16> -> !zames.tile<b, 4x8, bf16>
  // CHECK: zames.load {{.*}} : memref<4x8xf32> -> !zames.tile<a, 4x8, tf32>
  %f = zames.load %f32 : memref<4x8xf32> -> !zames.tile<a, 4x8, tf32>
  // CHECK: zames.load {{.*}} : memref<4x8xf32> -> !zames.tile<c, 4x8, fp32>
  %g = zames.load %f32 : memref<4x8xf32> -> !zames.tile<c, 4x8, fp32>
  zames.store %a, %i4 : !zames.tile<a, 4x8, i4>, memref<4x8xi4>
  zames.store %b, %u4 : !zames.tile<a, 4x8, u4>, memref<4x8xui4>
  zames.store %c, %u8 : !zames.tile<b, 4x8, u8>, memref<4x8xui8>
  zames.store %d, %f16 : !zames.tile<a, 4x8, fp16>, memref<4x8xf16>
  zames.store %e, %bf16 : !zames.tile<b, 4x8, bf16>, memref<4x8xbf16>
  zames.store %f, %f32 : !zames.tile<a, 4x8, tf32>, memref<4x8xf32>
  zames.store %g, %f32 : !zames.tile<c, 4x8, fp32>, memref<4x8xf32>
  return
}

// These tokens have no one-to-one builtin storage mapping at L1. Their native
// lowering support is intentionally not implied by representability here.
// CHECK-LABEL: func.func @historical_packed_storage
func.func @historical_packed_storage(%view: memref<4x8xi8>) {
  // CHECK: zames.load {{.*}} -> !zames.tile<a, 4x8, nvfp4>
  %a = zames.load %view : memref<4x8xi8> -> !zames.tile<a, 4x8, nvfp4>
  // CHECK: zames.load {{.*}} -> !zames.tile<b, 4x8, mxfp4>
  %b = zames.load %view : memref<4x8xi8> -> !zames.tile<b, 4x8, mxfp4>
  // CHECK: zames.load {{.*}} -> !zames.tile<a, 4x8, fp2pack4>
  %c = zames.load %view : memref<4x8xi8> -> !zames.tile<a, 4x8, fp2pack4>
  // CHECK: zames.load {{.*}} -> !zames.tile<b, 4x8, fp2pack5>
  %d = zames.load %view : memref<4x8xi8> -> !zames.tile<b, 4x8, fp2pack5>
  zames.store %a, %view : !zames.tile<a, 4x8, nvfp4>, memref<4x8xi8>
  zames.store %b, %view : !zames.tile<b, 4x8, mxfp4>, memref<4x8xi8>
  zames.store %c, %view : !zames.tile<a, 4x8, fp2pack4>, memref<4x8xi8>
  zames.store %d, %view : !zames.tile<b, 4x8, fp2pack5>, memref<4x8xi8>
  return
}
