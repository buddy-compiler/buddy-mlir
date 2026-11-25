// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// CHECK-LABEL: func.func @static_mma
func.func @static_mma(%a: !xsai_ame.tile<a, 4x16, i8>,
                      %b: !xsai_ame.tile<b, 8x16, i8>,
                      %c: !xsai_ame.tile<c, 4x8, i32>) -> !xsai_ame.tile<c, 4x8, i32> {
  // CHECK: xsai_ame.mma {{.*}} : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<c, 4x8, i32>{{$}}
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<c, 4x8, i32>
  return %r : !xsai_ame.tile<c, 4x8, i32>
}

// CHECK-LABEL: func.func @dynamic_mma
func.func @dynamic_mma(%a: !xsai_ame.tile<a, ?x?, fp16>,
                       %b: !xsai_ame.tile<b, ?x?, fp16>,
                       %c: !xsai_ame.tile<c, ?x?, fp32>,
                       %m: i64, %n: i64, %k: i64) -> !xsai_ame.tile<c, ?x?, fp32> {
  // CHECK: xsai_ame.mma {{.*}}[{{.*}}] : !xsai_ame.tile<a, ?x?, fp16>, !xsai_ame.tile<b, ?x?, fp16>, !xsai_ame.tile<c, ?x?, fp32>{{$}}
  %r = xsai_ame.mma %a, %b, %c[%m, %n, %k] : !xsai_ame.tile<a, ?x?, fp16>, !xsai_ame.tile<b, ?x?, fp16>, !xsai_ame.tile<c, ?x?, fp32>
  return %r : !xsai_ame.tile<c, ?x?, fp32>
}

// CHECK-LABEL: func.func @mixed_mma
func.func @mixed_mma(%a: !xsai_ame.tile<a, ?x16, bf16>,
                     %b: !xsai_ame.tile<b, 8x?, bf16>,
                     %c: !xsai_ame.tile<c, 4x8, fp32>,
                     %m: i64) -> !xsai_ame.tile<c, 4x8, fp32> {
  // Only A.M is dynamic among A.M/B.N/A.K: B.K does not add a size operand.
  // CHECK: xsai_ame.mma {{.*}}[{{.*}}] : !xsai_ame.tile<a, ?x16, bf16>, !xsai_ame.tile<b, 8x?, bf16>, !xsai_ame.tile<c, 4x8, fp32>{{$}}
  %r = xsai_ame.mma %a, %b, %c[%m] : !xsai_ame.tile<a, ?x16, bf16>, !xsai_ame.tile<b, 8x?, bf16>, !xsai_ame.tile<c, 4x8, fp32>
  return %r : !xsai_ame.tile<c, 4x8, fp32>
}

// CHECK-LABEL: func.func @dynamic_accumulator_static_inputs
func.func @dynamic_accumulator_static_inputs(%a: !xsai_ame.tile<a, 4x16, i8>,
                                            %b: !xsai_ame.tile<b, 8x16, i8>,
                                            %c: !xsai_ame.tile<c, ?x?, i32>) -> !xsai_ame.tile<c, ?x?, i32> {
  // Dynamic C dimensions alone do not add MMA size operands.
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<c, ?x?, i32>
  return %r : !xsai_ame.tile<c, ?x?, i32>
}

// CHECK-LABEL: func.func @mma_n_k_sizes
func.func @mma_n_k_sizes(%a: !xsai_ame.tile<a, 4x?, fp8e4m3>,
                         %b: !xsai_ame.tile<b, ?x16, fp8e4m3>,
                         %c: !xsai_ame.tile<c, 4x?, fp32>,
                         %n: i64, %k: i64) -> !xsai_ame.tile<c, 4x?, fp32> {
  %r = xsai_ame.mma %a, %b, %c[%n, %k] : !xsai_ame.tile<a, 4x?, fp8e4m3>, !xsai_ame.tile<b, ?x16, fp8e4m3>, !xsai_ame.tile<c, 4x?, fp32>
  return %r : !xsai_ame.tile<c, 4x?, fp32>
}

// CHECK-LABEL: func.func @zero_tiles
func.func @zero_tiles(%m: i64, %n: i64) -> (!xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, ?x16, fp16>, !xsai_ame.tile<c, ?x?, fp32>) {
  // CHECK: xsai_ame.mzero {test.tag = "static"} : !xsai_ame.tile<a, 4x16, i8>
  %a = xsai_ame.mzero {test.tag = "static"} : !xsai_ame.tile<a, 4x16, i8>
  // CHECK: xsai_ame.mzero[{{.*}}] : !xsai_ame.tile<b, ?x16, fp16>
  %b = xsai_ame.mzero[%m] : !xsai_ame.tile<b, ?x16, fp16>
  // CHECK: xsai_ame.mzero[{{.*}}] {test.tag = "dynamic"} : !xsai_ame.tile<c, ?x?, fp32>
  %c = xsai_ame.mzero[%m, %n] {test.tag = "dynamic"} : !xsai_ame.tile<c, ?x?, fp32>
  return %a, %b, %c : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, ?x16, fp16>, !xsai_ame.tile<c, ?x?, fp32>
}
