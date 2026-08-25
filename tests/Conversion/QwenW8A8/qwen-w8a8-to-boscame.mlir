// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame | FileCheck %s

module {
  func.func @decode_minimum(
      %input: memref<1x1024xf32>,
      %xq: memref<1x1024xi8>,
      %xs: memref<1x1xf32>,
      %wq: memref<1x1x64x1024xi8>,
      %ws: memref<1x64xf32>,
      %output: memref<1x64xf32>) {
    "bosc_ame.quantize_per_group"(%input, %xq, %xs)
        {group_size = 1024 : i64} :
        (memref<1x1024xf32>, memref<1x1024xi8>, memref<1x1xf32>) -> ()
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<1x1x64x1024xi8>, memref<1x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}

// CHECK-DAG: memref.global "private" @__buddy_qwen_w8a8_scratch_f32
// CHECK-DAG: memref.global "private" @__buddy_qwen_w8a8_zero_f32 : memref<16x64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
// CHECK-LABEL: func.func @decode_minimum
// CHECK-NOT: bosc_ame.quantize_per_group
// CHECK-NOT: bosc_ame.w8a8_linear
// CHECK: scf.for
// CHECK: math.absf
// CHECK: arith.maximumf
// CHECK-DAG: bosc_ame.msettilem
// CHECK-DAG: bosc_ame.msettilen
// CHECK-DAG: bosc_ame.msettilek
// CHECK: bosc_ame.mlce32.m 0
// CHECK: bosc_ame.mlce32.m 1
// CHECK: bosc_ame.mlce32.m 2
// CHECK: bosc_ame.mlce32.m 3
// CHECK: scf.for
// CHECK: bosc_ame.mlae8.m
// CHECK: bosc_ame.mlbe8.m 4
// CHECK: bosc_ame.mlbe8.m 5
// CHECK: bosc_ame.mqma.b.mm 0, 0, 4
// CHECK: bosc_ame.mlbe8.m 6
// CHECK: bosc_ame.mqma.b.mm 1, 0, 5
// CHECK: bosc_ame.mlbe8.m 7
// CHECK: bosc_ame.mqma.b.mm 2, 0, 6
// CHECK: bosc_ame.mqma.b.mm 3, 0, 7
// CHECK: bosc_ame.msce32.m 0
// CHECK: bosc_ame.msce32.m 1
// CHECK: bosc_ame.msce32.m 2
// CHECK: bosc_ame.msce32.m 3
// CHECK: arith.mulf
// CHECK: arith.addf
