// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// All historical tokens remain representable, independently of native lowering.
// CHECK-LABEL: func.func private @historical_tokens(
// CHECK-SAME: !zames.tile<a, 4x8, i4>
// CHECK-SAME: !zames.tile<b, 8x4, u4>
// CHECK-SAME: !zames.tile<a, ?x8, i8>
// CHECK-SAME: !zames.tile<b, 8x?, u8>
// CHECK-SAME: !zames.tile<c, ?x?, i32>
// CHECK-SAME: !zames.tile<a, 4x8, nvfp4>
// CHECK-SAME: !zames.tile<b, 8x4, mxfp4>
// CHECK-SAME: !zames.tile<a, 4x8, fp8e5m2>
// CHECK-SAME: !zames.tile<b, 8x4, fp8e4m3>
// CHECK-SAME: !zames.tile<a, 4x8, fp16>
// CHECK-SAME: !zames.tile<b, 8x4, bf16>
// CHECK-SAME: !zames.tile<a, 4x8, tf32>
// CHECK-SAME: !zames.tile<c, 4x8, fp32>
// CHECK-SAME: !zames.tile<a, 4x8, fp2pack4>
// CHECK-SAME: !zames.tile<b, 8x4, fp2pack5>
func.func private @historical_tokens(
  !zames.tile<a, 4x8, i4>, !zames.tile<b, 8x4, u4>,
  !zames.tile<a, ?x8, i8>, !zames.tile<b, 8x?, u8>,
  !zames.tile<c, ?x?, i32>, !zames.tile<a, 4x8, nvfp4>,
  !zames.tile<b, 8x4, mxfp4>, !zames.tile<a, 4x8, fp8e5m2>,
  !zames.tile<b, 8x4, fp8e4m3>, !zames.tile<a, 4x8, fp16>,
  !zames.tile<b, 8x4, bf16>, !zames.tile<a, 4x8, tf32>,
  !zames.tile<c, 4x8, fp32>, !zames.tile<a, 4x8, fp2pack4>,
  !zames.tile<b, 8x4, fp2pack5>)

// CHECK-LABEL: func.func private @sync_type(!zames.sync) -> !zames.sync
func.func private @sync_type(!zames.sync) -> !zames.sync
