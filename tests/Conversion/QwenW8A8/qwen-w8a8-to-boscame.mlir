// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame='target=qwen3-fpga' | FileCheck %s
// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame='target=qwen3-fpga profile-phases' | FileCheck %s --check-prefix=PROFILE

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
// CHECK-DAG: memref.global "private" @__buddy_qwen_w8a8_zero_f32 : memref<32x64xf32> = dense<0.000000e+00> alignment = 64
// CHECK-DAG: memref.global "private" @__buddy_qwen_w8a8_tail_activation_i8 : memref<16x64xi8> = uninitialized alignment = 64
// CHECK-LABEL: func.func @decode_minimum
// CHECK-NOT: bosc_ame.quantize_per_group
// CHECK-NOT: bosc_ame.w8a8_linear
// CHECK: scf.for
// CHECK: math.absf
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: bosc_ame.msettilem %[[ONE]]
// CHECK: bosc_ame.msettilen %[[ONE]]
// CHECK: bosc_ame.msettilek %[[ONE]]
// CHECK: bosc_ame.mlae8.m
// CHECK: bosc_ame.mlbte8.m
// CHECK: bosc_ame.mqma.b.mm %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: bosc_ame.mlce32.m
// CHECK: bosc_ame.msce32.m
// CHECK-NEXT: llvm.fence seq_cst
// CHECK: bosc_ame.msettilen
// CHECK: bosc_ame.msettilek
// CHECK: bosc_ame.msettilem %[[ONE]]
// CHECK: bosc_ame.mlce32.m
// CHECK: bosc_ame.mlce32.m
// CHECK: bosc_ame.mlce32.m
// CHECK: bosc_ame.mlce32.m
// CHECK: scf.for
// CHECK: bosc_ame.mlae8.m
// CHECK: bosc_ame.mlbe8.m
// CHECK: bosc_ame.mlbe8.m
// CHECK: bosc_ame.mqma.b.mm %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: bosc_ame.mlbe8.m
// CHECK: bosc_ame.mqma.b.mm %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: bosc_ame.mlbe8.m
// CHECK: bosc_ame.mqma.b.mm %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: bosc_ame.mqma.b.mm %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: bosc_ame.msce32.m
// CHECK: bosc_ame.msce32.m
// CHECK: bosc_ame.msce32.m
// CHECK: bosc_ame.msce32.m
// CHECK-NEXT: llvm.fence seq_cst
// CHECK-NOT: arith.sitofp
// CHECK: func.call @buddy_w8a8_rvv_accumulate_n64

// PROFILE-LABEL: func.func @decode_minimum
// PROFILE: %[[QUANT_START:.*]] = arith.constant 252 : i64
// PROFILE: call @buddyTraceCycleStartPath(%[[QUANT_START]],
// PROFILE: %[[QUANT_END:.*]] = arith.constant 252 : i64
// PROFILE: call @buddyTraceCycleEndPath(%[[QUANT_END]],
// PROFILE: %[[LINEAR_START:.*]] = arith.constant 251 : i64
// PROFILE: call @buddyTraceCycleStartPath(%[[LINEAR_START]],
// PROFILE: %[[LINEAR_END:.*]] = arith.constant 251 : i64
// PROFILE: call @buddyTraceCycleEndPath(%[[LINEAR_END]],
