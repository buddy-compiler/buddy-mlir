// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// All historical tokens remain representable, independently of native lowering.
// CHECK-LABEL: func.func private @historical_tokens(
// CHECK-SAME: !xsai_ame.tile<a, 4x8, i4>
// CHECK-SAME: !xsai_ame.tile<b, 8x4, u4>
// CHECK-SAME: !xsai_ame.tile<a, ?x8, i8>
// CHECK-SAME: !xsai_ame.tile<b, 8x?, u8>
// CHECK-SAME: !xsai_ame.tile<c, ?x?, i32>
// CHECK-SAME: !xsai_ame.tile<a, 4x8, nvfp4>
// CHECK-SAME: !xsai_ame.tile<b, 8x4, mxfp4>
// CHECK-SAME: !xsai_ame.tile<a, 4x8, fp8e5m2>
// CHECK-SAME: !xsai_ame.tile<b, 8x4, fp8e4m3>
// CHECK-SAME: !xsai_ame.tile<a, 4x8, fp16>
// CHECK-SAME: !xsai_ame.tile<b, 8x4, bf16>
// CHECK-SAME: !xsai_ame.tile<a, 4x8, tf32>
// CHECK-SAME: !xsai_ame.tile<c, 4x8, fp32>
// CHECK-SAME: !xsai_ame.tile<a, 4x8, fp2pack4>
// CHECK-SAME: !xsai_ame.tile<b, 8x4, fp2pack5>
func.func private @historical_tokens(
  !xsai_ame.tile<a, 4x8, i4>, !xsai_ame.tile<b, 8x4, u4>,
  !xsai_ame.tile<a, ?x8, i8>, !xsai_ame.tile<b, 8x?, u8>,
  !xsai_ame.tile<c, ?x?, i32>, !xsai_ame.tile<a, 4x8, nvfp4>,
  !xsai_ame.tile<b, 8x4, mxfp4>, !xsai_ame.tile<a, 4x8, fp8e5m2>,
  !xsai_ame.tile<b, 8x4, fp8e4m3>, !xsai_ame.tile<a, 4x8, fp16>,
  !xsai_ame.tile<b, 8x4, bf16>, !xsai_ame.tile<a, 4x8, tf32>,
  !xsai_ame.tile<c, 4x8, fp32>, !xsai_ame.tile<a, 4x8, fp2pack4>,
  !xsai_ame.tile<b, 8x4, fp2pack5>)

// CHECK-LABEL: func.func private @sync_type(!xsai_ame.sync) -> !xsai_ame.sync
func.func private @sync_type(!xsai_ame.sync) -> !xsai_ame.sync
