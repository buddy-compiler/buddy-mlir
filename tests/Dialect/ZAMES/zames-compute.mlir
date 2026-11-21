// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// CHECK-LABEL: func.func @static_mma
func.func @static_mma(%a: !zames.tile<a, 4x16, i8>,
                      %b: !zames.tile<b, 8x16, i8>,
                      %c: !zames.tile<c, 4x8, i32>) -> !zames.tile<c, 4x8, i32> {
  // CHECK: zames.mma {{.*}} : !zames.tile<a, 4x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<c, 4x8, i32>{{$}}
  %r = zames.mma %a, %b, %c : !zames.tile<a, 4x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<c, 4x8, i32>
  return %r : !zames.tile<c, 4x8, i32>
}

// CHECK-LABEL: func.func @dynamic_mma
func.func @dynamic_mma(%a: !zames.tile<a, ?x?, fp16>,
                       %b: !zames.tile<b, ?x?, fp16>,
                       %c: !zames.tile<c, ?x?, fp32>,
                       %m: i64, %n: i64, %k: i64) -> !zames.tile<c, ?x?, fp32> {
  // CHECK: zames.mma {{.*}}[{{.*}}] : !zames.tile<a, ?x?, fp16>, !zames.tile<b, ?x?, fp16>, !zames.tile<c, ?x?, fp32>{{$}}
  %r = zames.mma %a, %b, %c[%m, %n, %k] : !zames.tile<a, ?x?, fp16>, !zames.tile<b, ?x?, fp16>, !zames.tile<c, ?x?, fp32>
  return %r : !zames.tile<c, ?x?, fp32>
}

// CHECK-LABEL: func.func @mixed_mma
func.func @mixed_mma(%a: !zames.tile<a, ?x16, bf16>,
                     %b: !zames.tile<b, 8x?, bf16>,
                     %c: !zames.tile<c, 4x8, fp32>,
                     %m: i64) -> !zames.tile<c, 4x8, fp32> {
  // Only A.M is dynamic among A.M/B.N/A.K: B.K does not add a size operand.
  // CHECK: zames.mma {{.*}}[{{.*}}] : !zames.tile<a, ?x16, bf16>, !zames.tile<b, 8x?, bf16>, !zames.tile<c, 4x8, fp32>{{$}}
  %r = zames.mma %a, %b, %c[%m] : !zames.tile<a, ?x16, bf16>, !zames.tile<b, 8x?, bf16>, !zames.tile<c, 4x8, fp32>
  return %r : !zames.tile<c, 4x8, fp32>
}

// CHECK-LABEL: func.func @dynamic_accumulator_static_inputs
func.func @dynamic_accumulator_static_inputs(%a: !zames.tile<a, 4x16, i8>,
                                            %b: !zames.tile<b, 8x16, i8>,
                                            %c: !zames.tile<c, ?x?, i32>) -> !zames.tile<c, ?x?, i32> {
  // Dynamic C dimensions alone do not add MMA size operands.
  %r = zames.mma %a, %b, %c : !zames.tile<a, 4x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<c, ?x?, i32>
  return %r : !zames.tile<c, ?x?, i32>
}

// CHECK-LABEL: func.func @mma_n_k_sizes
func.func @mma_n_k_sizes(%a: !zames.tile<a, 4x?, fp8e4m3>,
                         %b: !zames.tile<b, ?x16, fp8e4m3>,
                         %c: !zames.tile<c, 4x?, fp32>,
                         %n: i64, %k: i64) -> !zames.tile<c, 4x?, fp32> {
  %r = zames.mma %a, %b, %c[%n, %k] : !zames.tile<a, 4x?, fp8e4m3>, !zames.tile<b, ?x16, fp8e4m3>, !zames.tile<c, 4x?, fp32>
  return %r : !zames.tile<c, 4x?, fp32>
}

// CHECK-LABEL: func.func @zero_tiles
func.func @zero_tiles(%m: i64, %n: i64) -> (!zames.tile<a, 4x16, i8>, !zames.tile<b, ?x16, fp16>, !zames.tile<c, ?x?, fp32>) {
  // CHECK: zames.mzero {test.tag = "static"} : !zames.tile<a, 4x16, i8>
  %a = zames.mzero {test.tag = "static"} : !zames.tile<a, 4x16, i8>
  // CHECK: zames.mzero[{{.*}}] : !zames.tile<b, ?x16, fp16>
  %b = zames.mzero[%m] : !zames.tile<b, ?x16, fp16>
  // CHECK: zames.mzero[{{.*}}] {test.tag = "dynamic"} : !zames.tile<c, ?x?, fp32>
  %c = zames.mzero[%m, %n] {test.tag = "dynamic"} : !zames.tile<c, ?x?, fp32>
  return %a, %b, %c : !zames.tile<a, 4x16, i8>, !zames.tile<b, ?x16, fp16>, !zames.tile<c, ?x?, fp32>
}
