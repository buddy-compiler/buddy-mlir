// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame | FileCheck %s

module {
  func.func @prefill_m8(
      %xq: memref<8x1024xi8>, %xs: memref<8x1xf32>,
      %wq: memref<1x1x64x1024xi8>, %ws: memref<1x64xf32>,
      %output: memref<8x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<8x1024xi8>, memref<8x1xf32>,
         memref<1x1x64x1024xi8>, memref<1x64xf32>,
         memref<8x64xf32>) -> ()
    return
  }

  func.func @prefill_m32(
      %xq: memref<32x1024xi8>, %xs: memref<32x1xf32>,
      %wq: memref<1x1x64x1024xi8>, %ws: memref<1x64xf32>,
      %output: memref<32x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<32x1024xi8>, memref<32x1xf32>,
         memref<1x1x64x1024xi8>, memref<1x64xf32>,
         memref<32x64xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @prefill_m8
// CHECK-NOT: bosc_ame.w8a8_linear
// CHECK: %[[M8:.*]] = arith.constant 8 : i64
// CHECK: bosc_ame.msettilem %[[M8]]
// CHECK: bosc_ame.mlae8.m
// CHECK: bosc_ame.mqma.b.mm
// CHECK: bosc_ame.msce32.m

// CHECK-LABEL: func.func @prefill_m32
// CHECK-NOT: bosc_ame.w8a8_linear
// CHECK: %[[M16:.*]] = arith.constant 16 : i64
// CHECK: scf.for
// CHECK: bosc_ame.msettilem %[[M16]]
// CHECK: bosc_ame.mlae8.m
// CHECK: bosc_ame.mqma.b.mm
// CHECK: bosc_ame.msce32.m
