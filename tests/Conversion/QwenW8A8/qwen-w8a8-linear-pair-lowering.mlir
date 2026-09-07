// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame -o %t.fast
// RUN: FileCheck %s --input-file=%t.fast --check-prefix=FAST
// RUN: FileCheck %s --input-file=%t.fast --check-prefix=NCOUNT
// RUN: FileCheck %s --input-file=%t.fast --check-prefix=KCOUNT
// RUN: FileCheck %s --input-file=%t.fast --check-prefix=SYNC
// RUN: FileCheck %s --input-file=%t.fast --check-prefix=PREFILL
// RUN: FileCheck %s --input-file=%t.fast --check-prefix=UNKNOWN
// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame="scalar-fallback=true" -o %t.scalar
// RUN: FileCheck %s --input-file=%t.scalar --check-prefix=SCALAR

module {
  func.func @pair_fast(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq0: memref<48x1x64x1024xi8>, %ws0: memref<1x3072xf32>,
      %wq1: memref<48x1x64x1024xi8>, %ws1: memref<1x3072xf32>) {
    %out0 = memref.alloc() : memref<1x3072xf32>
    %out1 = memref.alloc() : memref<1x3072xf32>
    "bosc_ame.w8a8_linear_pair"(
        %xq, %xs, %wq0, %ws0, %wq1, %ws1, %out0, %out1)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<1x3072xf32>, memref<1x3072xf32>) -> ()
    memref.dealloc %out0 : memref<1x3072xf32>
    memref.dealloc %out1 : memref<1x3072xf32>
    return
  }

  // Even with provably distinct fresh outputs, prefill deliberately retains
  // the two complete schedules.  The shared-clear variant regressed S=22 on
  // FPGA and the handwritten GS1024 pair takes the same fallback.
  func.func @pair_prefill_fallback(
      %xq: memref<22x1024xi8>, %xs: memref<22x1xf32>,
      %wq0: memref<48x1x64x1024xi8>, %ws0: memref<1x3072xf32>,
      %wq1: memref<48x1x64x1024xi8>, %ws1: memref<1x3072xf32>) {
    %out0 = memref.alloc() : memref<22x3072xf32>
    %out1 = memref.alloc() : memref<22x3072xf32>
    "bosc_ame.w8a8_linear_pair"(
        %xq, %xs, %wq0, %ws0, %wq1, %ws1, %out0, %out1)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<22x1024xi8>, memref<22x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<22x3072xf32>, memref<22x3072xf32>) -> ()
    memref.dealloc %out0 : memref<22x3072xf32>
    memref.dealloc %out1 : memref<22x3072xf32>
    return
  }

  // Two distinct block arguments are not a proof of non-aliasing at runtime.
  // The lowering must conservatively retain both complete preambles.
  func.func @pair_unknown_output_alias(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq0: memref<48x1x64x1024xi8>, %ws0: memref<1x3072xf32>,
      %wq1: memref<48x1x64x1024xi8>, %ws1: memref<1x3072xf32>,
      %out0: memref<1x3072xf32>, %out1: memref<1x3072xf32>) {
    "bosc_ame.w8a8_linear_pair"(
        %xq, %xs, %wq0, %ws0, %wq1, %ws1, %out0, %out1)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<1x3072xf32>, memref<1x3072xf32>) -> ()
    return
  }

  // This control function has the same two projections but no pair op.  It
  // demonstrates the two independent preambles that the paired fast path
  // replaces with one.
  func.func @two_independent(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq0: memref<48x1x64x1024xi8>, %ws0: memref<1x3072xf32>,
      %wq1: memref<48x1x64x1024xi8>, %ws1: memref<1x3072xf32>,
      %out0: memref<1x3072xf32>, %out1: memref<1x3072xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq0, %ws0, %out0)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<1x3072xf32>) -> ()
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq1, %ws1, %out1)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<1x3072xf32>) -> ()
    return
  }
}

// Both semantic pair and temporary child ops must be fully consumed by the
// default conversion.
// FAST-LABEL: func.func @pair_fast
// FAST-NOT: bosc_ame.w8a8_linear_pair
// FAST-NOT: "bosc_ame.w8a8_linear"
// FAST: return

// One shared resync programs N/K first to 1 and then to the invariant 16/64.
// The independent control has two copies of each sequence.
// NCOUNT-LABEL: func.func @pair_fast
// NCOUNT-COUNT-2: bosc_ame.msettilen
// NCOUNT-NOT: bosc_ame.msettilen
// NCOUNT-LABEL: func.func @pair_prefill_fallback
// NCOUNT-COUNT-4: bosc_ame.msettilen
// NCOUNT-NOT: bosc_ame.msettilen
// NCOUNT-LABEL: func.func @pair_unknown_output_alias
// NCOUNT-COUNT-4: bosc_ame.msettilen
// NCOUNT-NOT: bosc_ame.msettilen
// NCOUNT-LABEL: func.func @two_independent
// NCOUNT-COUNT-4: bosc_ame.msettilen
// NCOUNT-NOT: bosc_ame.msettilen

// KCOUNT-LABEL: func.func @pair_fast
// KCOUNT-COUNT-2: bosc_ame.msettilek
// KCOUNT-NOT: bosc_ame.msettilek
// KCOUNT-LABEL: func.func @pair_prefill_fallback
// KCOUNT-COUNT-4: bosc_ame.msettilek
// KCOUNT-NOT: bosc_ame.msettilek
// KCOUNT-LABEL: func.func @pair_unknown_output_alias
// KCOUNT-COUNT-4: bosc_ame.msettilek
// KCOUNT-NOT: bosc_ame.msettilek
// KCOUNT-LABEL: func.func @two_independent
// KCOUNT-COUNT-4: bosc_ame.msettilek
// KCOUNT-NOT: bosc_ame.msettilek

// mlbte8 is unique to the minimal operation-boundary resync sequence.
// SYNC-LABEL: func.func @pair_fast
// SYNC-COUNT-1: bosc_ame.mlbte8.m
// SYNC-NOT: bosc_ame.mlbte8.m
// SYNC-LABEL: func.func @pair_prefill_fallback
// SYNC-COUNT-2: bosc_ame.mlbte8.m
// SYNC-NOT: bosc_ame.mlbte8.m
// SYNC-LABEL: func.func @pair_unknown_output_alias
// SYNC-COUNT-2: bosc_ame.mlbte8.m
// SYNC-NOT: bosc_ame.mlbte8.m

// PREFILL-LABEL: func.func @pair_prefill_fallback
// PREFILL-NOT: bosc_ame.w8a8_linear_pair
// PREFILL-NOT: "bosc_ame.w8a8_linear"
// PREFILL-COUNT-2: bosc_ame.mlbte8.m
// PREFILL: return

// UNKNOWN-LABEL: func.func @pair_unknown_output_alias
// UNKNOWN-COUNT-2: bosc_ame.mlbte8.m
// UNKNOWN-NOT: bosc_ame.mlbte8.m
// UNKNOWN: return
// SYNC-LABEL: func.func @two_independent
// SYNC-COUNT-2: bosc_ame.mlbte8.m
// SYNC-NOT: bosc_ame.mlbte8.m

// The diagnostic scalar mode first decomposes the pair and then lowers both
// child linears through the established scalar/N64 fallback.
// SCALAR-LABEL: func.func @pair_fast
// SCALAR-NOT: bosc_ame.w8a8_linear_pair
// SCALAR-NOT: "bosc_ame.w8a8_linear"
// SCALAR-COUNT-2: bosc_ame.mlbte8.m
// SCALAR: memref.load
// SCALAR: arith.mulf
// SCALAR: arith.addf
// SCALAR: memref.store
// SCALAR-NOT: buddy_w8a8_rvv_accumulate_n64
// SCALAR: return
