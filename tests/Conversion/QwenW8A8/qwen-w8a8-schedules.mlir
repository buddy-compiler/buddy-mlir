// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame | FileCheck %s
// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame='scalar-fallback' | FileCheck %s --check-prefix=SCALAR
// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame='profile-phases' | FileCheck %s --check-prefix=PROFILE

module {
  func.func @decode_pair_gs512(
      %xq: memref<1x512xi8>, %xs: memref<1x1xf32>,
      %wq: memref<2x1x64x512xi8>, %ws: memref<1x128xf32>,
      %output: memref<1x128xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 512 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x512xi8>, memref<1x1xf32>, memref<2x1x64x512xi8>,
         memref<1x128xf32>, memref<1x128xf32>) -> ()
    return
  }

  func.func @decode_n64_fallback_gs1024(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq: memref<3x1x64x1024xi8>, %ws: memref<1x192xf32>,
      %output: memref<1x192xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>, memref<3x1x64x1024xi8>,
         memref<1x192xf32>, memref<1x192xf32>) -> ()
    return
  }

  func.func @prefill_t16_gs512(
      %xq: memref<16x512xi8>, %xs: memref<16x1xf32>,
      %wq: memref<1x1x64x512xi8>, %ws: memref<1x64xf32>,
      %output: memref<16x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 512 : i64, weight_layout = "ame_outblk64"} :
        (memref<16x512xi8>, memref<16x1xf32>, memref<1x1x64x512xi8>,
         memref<1x64xf32>, memref<16x64xf32>) -> ()
    return
  }

  func.func @prefill_t22_gs512(
      %xq: memref<22x512xi8>, %xs: memref<22x1xf32>,
      %wq: memref<1x1x64x512xi8>, %ws: memref<1x64xf32>,
      %output: memref<22x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 512 : i64, weight_layout = "ame_outblk64"} :
        (memref<22x512xi8>, memref<22x1xf32>, memref<1x1x64x512xi8>,
         memref<1x64xf32>, memref<22x64xf32>) -> ()
    return
  }

  func.func @prefill_t32_gs1024(
      %xq: memref<32x1024xi8>, %xs: memref<32x1xf32>,
      %wq: memref<1x1x64x1024xi8>, %ws: memref<1x64xf32>,
      %output: memref<32x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<32x1024xi8>, memref<32x1xf32>, memref<1x1x64x1024xi8>,
         memref<1x64xf32>, memref<32x64xf32>) -> ()
    return
  }

  func.func @prefill_t128_gs512(
      %xq: memref<128x512xi8>, %xs: memref<128x1xf32>,
      %wq: memref<1x1x64x512xi8>, %ws: memref<1x64xf32>,
      %output: memref<128x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 512 : i64, weight_layout = "ame_outblk64"} :
        (memref<128x512xi8>, memref<128x1xf32>, memref<1x1x64x512xi8>,
         memref<1x64xf32>, memref<128x64xf32>) -> ()
    return
  }

  func.func @prefill_t8_gs512(
      %xq: memref<8x512xi8>, %xs: memref<8x1xf32>,
      %wq: memref<1x1x64x512xi8>, %ws: memref<1x64xf32>,
      %output: memref<8x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 512 : i64, weight_layout = "ame_outblk64"} :
        (memref<8x512xi8>, memref<8x1xf32>, memref<1x1x64x512xi8>,
         memref<1x64xf32>, memref<8x64xf32>) -> ()
    return
  }

  func.func @prefill_t64_gs1024(
      %xq: memref<64x1024xi8>, %xs: memref<64x1xf32>,
      %wq: memref<1x1x64x1024xi8>, %ws: memref<1x64xf32>,
      %output: memref<64x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<64x1024xi8>, memref<64x1xf32>, memref<1x1x64x1024xi8>,
         memref<1x64xf32>, memref<64x64xf32>) -> ()
    return
  }
}

// CHECK-DAG: memref.global "private" @__buddy_qwen_w8a8_scratch_f32 : memref<32x64xf32>
// CHECK-DAG: memref.global "private" @__buddy_qwen_w8a8_zero_f32 : memref<32x64xf32>

// Decode pairs adjacent OUTBLK64 blocks and uses all eight accumulators.
// CHECK-LABEL: func.func @decode_pair_gs512
// SCALAR-LABEL: func.func @decode_pair_gs512
// SCALAR: bosc_ame.mqma.b.mm 3, 0, 7
// SCALAR-NOT: bosc_ame.mqma.b.mm 7,
// SCALAR-NOT: vector.load
// PROFILE-LABEL: func.func @decode_pair_gs512
// PROFILE: call @buddyTraceCycleStartPath(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64) -> ()
// PROFILE: bosc_ame.mqma.b.mm 7, 0, 7
// PROFILE: call @buddyTraceCycleEndPath(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64) -> ()
// CHECK: bosc_ame.msettilem
// CHECK: scf.for
// CHECK: scf.for
// CHECK: bosc_ame.mlce32.m 7
// CHECK: scf.for {{.*}} to %{{.*}} step %{{.*}} {
// CHECK: bosc_ame.mlae8.m 0
// CHECK: bosc_ame.mqma.b.mm 3, 0, 7
// CHECK: bosc_ame.mlbe8.m 4
// CHECK: bosc_ame.mqma.b.mm 4, 0, 4
// CHECK: bosc_ame.mqma.b.mm 7, 0, 7
// CHECK: bosc_ame.msce32.m 7
// CHECK-NEXT: llvm.fence seq_cst
// CHECK: func.call @buddy_w8a8_rvv_accumulate_n64
// CHECK: func.call @buddy_w8a8_rvv_accumulate_n64

// Three N64 blocks use one paired N128 kernel followed by reliable 1A x 4B.
// CHECK-LABEL: func.func @decode_n64_fallback_gs1024
// CHECK: bosc_ame.mqma.b.mm 7, 0, 7
// CHECK: bosc_ame.msce32.m 7
// CHECK: bosc_ame.mlce32.m 0
// CHECK: bosc_ame.mlce32.m 3
// CHECK: bosc_ame.mqma.b.mm 3, 0, 7
// CHECK: bosc_ame.msce32.m 3

// T=16 remains the validated 1A x 4B schedule.
// CHECK-LABEL: func.func @prefill_t16_gs512
// CHECK: %[[M16:.*]] = arith.constant 16 : i64
// CHECK: bosc_ame.msettilem %[[M16]]
// CHECK: bosc_ame.mlae8.m 0
// CHECK-NOT: bosc_ame.mlae8.m 2
// CHECK: bosc_ame.mqma.b.mm 3, 0, 7
// CHECK-NOT: bosc_ame.mqma.b.mm 4,

// T=22 is one exact M16 tile followed by an exact M6 tail.
// CHECK-LABEL: func.func @prefill_t22_gs512
// CHECK: bosc_ame.msettilem
// CHECK: bosc_ame.mqma.b.mm 3, 0, 7
// CHECK: %[[M6:.*]] = arith.constant 6 : i64
// CHECK: bosc_ame.msettilem %[[M6]]
// CHECK: bosc_ame.mqma.b.mm 3, 0, 7

// T=32 uses 2A x 4B: two A loads, four B loads and acc0..acc7.
// CHECK-LABEL: func.func @prefill_t32_gs1024
// CHECK: bosc_ame.msettilem
// CHECK: scf.for
// CHECK: bosc_ame.mlae8.m 0
// CHECK-NEXT: bosc_ame.mlae8.m 2
// CHECK: bosc_ame.mlbe8.m 4
// CHECK: bosc_ame.mlbe8.m 5
// CHECK: bosc_ame.mqma.b.mm 0, 0, 4
// CHECK: bosc_ame.mlbe8.m 6
// CHECK: bosc_ame.mqma.b.mm 4, 2, 4
// CHECK: bosc_ame.mlbe8.m 7
// CHECK: bosc_ame.mqma.b.mm 7, 2, 7
// CHECK: bosc_ame.msce32.m 7
// CHECK-NEXT: llvm.fence seq_cst

// T=128 executes the same 32-row body four times.
// CHECK-LABEL: func.func @prefill_t128_gs512
// CHECK: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK: scf.for {{.*}} to %[[FOUR]] step
// CHECK: bosc_ame.mlae8.m 0
// CHECK-NEXT: bosc_ame.mlae8.m 2
// CHECK: bosc_ame.mqma.b.mm 7, 2, 7

// T=8 uses an exact M8 1A x 4B tail.
// CHECK-LABEL: func.func @prefill_t8_gs512
// CHECK: %[[M8:.*]] = arith.constant 8 : i64
// CHECK: bosc_ame.msettilem %[[M8]]
// CHECK: bosc_ame.mlae8.m 0
// CHECK-NOT: bosc_ame.mlae8.m 2
// CHECK: bosc_ame.mqma.b.mm 3, 0, 7

// T=64 executes two 32-row 2A x 4B iterations.
// CHECK-LABEL: func.func @prefill_t64_gs1024
// CHECK: bosc_ame.msettilem
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-NEXT: scf.for {{.*}} to %[[TWO]] step
// CHECK: bosc_ame.mlae8.m 0
// CHECK-NEXT: bosc_ame.mlae8.m 2
// CHECK: bosc_ame.mqma.b.mm 7, 2, 7
