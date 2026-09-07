// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame | FileCheck %s --check-prefix=UNROLL-DIV
// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame='quantize-reciprocal=true' | FileCheck %s --check-prefix=RECIPROCAL
// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame='quantize-one-ahead=true' | FileCheck %s --check-prefix=ONE-AHEAD
// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame='quantize-unroll=false quantize-reciprocal=false' | FileCheck %s --check-prefix=FALLBACK
// RUN: not buddy-opt %s --lower-qwen-w8a8-to-boscame='quantize-reciprocal=true quantize-one-ahead=true' 2>&1 | FileCheck %s --check-prefix=CONFLICT

module {
  func.func @quantize_gs512(
      %input: memref<1x512xf32>,
      %quantized: memref<1x512xi8>,
      %scales: memref<1x1xf32>) {
    "bosc_ame.quantize_per_group"(%input, %quantized, %scales)
        {group_size = 512 : i64} :
        (memref<1x512xf32>, memref<1x512xi8>, memref<1x1xf32>) -> ()
    return
  }

  func.func @quantize_gs1024(
      %input: memref<1x1024xf32>,
      %quantized: memref<1x1024xi8>,
      %scales: memref<1x1xf32>) {
    "bosc_ame.quantize_per_group"(%input, %quantized, %scales)
        {group_size = 1024 : i64} :
        (memref<1x1024xf32>, memref<1x1024xi8>, memref<1x1xf32>) -> ()
    return
  }

  // Unsupported model group sizes deliberately retain the ordered scalar
  // lowering even when both fast-path options use their default value.
  func.func @quantize_gs256(
      %input: memref<1x256xf32>,
      %quantized: memref<1x256xi8>,
      %scales: memref<1x1xf32>) {
    "bosc_ame.quantize_per_group"(%input, %quantized, %scales)
        {group_size = 256 : i64} :
        (memref<1x256xf32>, memref<1x256xi8>, memref<1x1xf32>) -> ()
    return
  }

  // The pointer-only helper requires contiguous columns.  Non-unit inner
  // strides keep the inlined scalar fallback even when one-ahead is enabled.
  func.func @quantize_gs512_strided(
      %input: memref<1x512xf32, strided<[1024, 2], offset: ?>>,
      %quantized: memref<1x512xi8, strided<[1024, 2], offset: ?>>,
      %scales: memref<1x1xf32>) {
    "bosc_ame.quantize_per_group"(%input, %quantized, %scales)
        {group_size = 512 : i64} :
        (memref<1x512xf32, strided<[1024, 2], offset: ?>>,
         memref<1x512xi8, strided<[1024, 2], offset: ?>>,
         memref<1x1xf32>) -> ()
    return
  }
}

// RECIPROCAL-LABEL: func.func @quantize_gs512
// RECIPROCAL: %[[STEP512:.*]] = arith.constant 8 : index
// RECIPROCAL: scf.for {{.*}} step %[[STEP512]] iter_args
// RECIPROCAL-COUNT-8: math.absf
// RECIPROCAL: scf.yield
// RECIPROCAL: %[[SCALE512:.*]] = arith.select
// RECIPROCAL: memref.store %[[SCALE512]],
// RECIPROCAL: %[[INV512:.*]] = arith.divf {{.*}}, %[[SCALE512]] : f32
// RECIPROCAL: scf.for {{.*}} step %[[STEP512]] {
// RECIPROCAL-COUNT-8: arith.mulf {{.*}}, %[[INV512]] : f32
// RECIPROCAL-NOT: arith.divf
// RECIPROCAL: return

// RECIPROCAL-LABEL: func.func @quantize_gs1024
// RECIPROCAL: %[[STEP1024:.*]] = arith.constant 8 : index
// RECIPROCAL: scf.for {{.*}} step %[[STEP1024]] iter_args
// RECIPROCAL-COUNT-8: math.absf
// RECIPROCAL: scf.yield
// RECIPROCAL: %[[SCALE1024:.*]] = arith.select
// RECIPROCAL: memref.store %[[SCALE1024]],
// RECIPROCAL: %[[INV1024:.*]] = arith.divf {{.*}}, %[[SCALE1024]] : f32
// RECIPROCAL: scf.for {{.*}} step %[[STEP1024]] {
// RECIPROCAL-COUNT-8: arith.mulf {{.*}}, %[[INV1024]] : f32
// RECIPROCAL-NOT: arith.divf
// RECIPROCAL: return

// RECIPROCAL-LABEL: func.func @quantize_gs256
// RECIPROCAL: %[[ONE256:.*]] = arith.constant 1 : index
// RECIPROCAL: scf.for {{.*}} step %[[ONE256]] iter_args
// RECIPROCAL-COUNT-1: math.absf
// RECIPROCAL: scf.yield
// RECIPROCAL: %[[SCALE256:.*]] = arith.select
// RECIPROCAL: scf.for {{.*}} step %[[ONE256]] {
// RECIPROCAL: %[[VALUE256:.*]] = memref.load
// RECIPROCAL: arith.divf %[[VALUE256]], %[[SCALE256]] : f32
// RECIPROCAL: return

// UNROLL-DIV-LABEL: func.func @quantize_gs512
// UNROLL-DIV: %[[STEP512:.*]] = arith.constant 8 : index
// UNROLL-DIV: scf.for {{.*}} step %[[STEP512]] iter_args
// UNROLL-DIV: scf.yield
// UNROLL-DIV: %[[SCALE512:.*]] = arith.select
// UNROLL-DIV: scf.for {{.*}} step %[[STEP512]] {
// UNROLL-DIV-COUNT-8: arith.divf {{.*}}, %[[SCALE512]] : f32
// UNROLL-DIV-NOT: arith.mulf
// UNROLL-DIV: return

// FALLBACK-LABEL: func.func @quantize_gs512
// FALLBACK: %[[ONE512:.*]] = arith.constant 1 : index
// FALLBACK: scf.for {{.*}} step %[[ONE512]] iter_args
// FALLBACK-COUNT-1: math.absf
// FALLBACK: scf.yield
// FALLBACK: %[[FBSCALE512:.*]] = arith.select
// FALLBACK: scf.for {{.*}} step %[[ONE512]] {
// FALLBACK: %[[FBVALUE512:.*]] = memref.load
// FALLBACK: arith.divf %[[FBVALUE512]], %[[FBSCALE512]] : f32
// FALLBACK-NOT: arith.mulf
// FALLBACK: return

// ONE-AHEAD: func.func private @buddy_w8a8_quantize_write_one_ahead(i64, i64, i32, i64)
// ONE-AHEAD-LABEL: func.func @quantize_gs512
// ONE-AHEAD: scf.for {{.*}} step %{{.*}} iter_args
// ONE-AHEAD: %[[RAW512:.*]] = arith.divf
// ONE-AHEAD: %[[SCALE512:.*]] = arith.select
// ONE-AHEAD: memref.store %[[SCALE512]],
// ONE-AHEAD: %[[BITS512:.*]] = arith.bitcast %[[SCALE512]] : f32 to i32
// ONE-AHEAD: %[[COUNT512:.*]] = arith.constant 512 : i64
// ONE-AHEAD: func.call @buddy_w8a8_quantize_write_one_ahead({{.*}}, {{.*}}, %[[BITS512]], %[[COUNT512]])
// ONE-AHEAD-NOT: arith.mulf
// ONE-AHEAD: return

// ONE-AHEAD-LABEL: func.func @quantize_gs1024
// ONE-AHEAD: %[[RAW1024:.*]] = arith.divf
// ONE-AHEAD: %[[SCALE1024:.*]] = arith.select
// ONE-AHEAD: memref.store %[[SCALE1024]],
// ONE-AHEAD: %[[BITS1024:.*]] = arith.bitcast %[[SCALE1024]] : f32 to i32
// ONE-AHEAD: %[[COUNT1024:.*]] = arith.constant 1024 : i64
// ONE-AHEAD: func.call @buddy_w8a8_quantize_write_one_ahead({{.*}}, {{.*}}, %[[BITS1024]], %[[COUNT1024]])
// ONE-AHEAD: return

// ONE-AHEAD-LABEL: func.func @quantize_gs256
// ONE-AHEAD: scf.for
// ONE-AHEAD: arith.divf
// ONE-AHEAD-NOT: buddy_w8a8_quantize_write_one_ahead
// ONE-AHEAD: return

// ONE-AHEAD-LABEL: func.func @quantize_gs512_strided
// ONE-AHEAD: scf.for
// ONE-AHEAD: arith.divf
// ONE-AHEAD-NOT: buddy_w8a8_quantize_write_one_ahead
// ONE-AHEAD: return

// CONFLICT: error: quantize-reciprocal and quantize-one-ahead are mutually exclusive

// FALLBACK-LABEL: func.func @quantize_gs1024
// FALLBACK: %[[ONE1024:.*]] = arith.constant 1 : index
// FALLBACK: scf.for {{.*}} step %[[ONE1024]] iter_args
// FALLBACK-COUNT-1: math.absf
// FALLBACK: scf.yield
// FALLBACK: %[[FBSCALE1024:.*]] = arith.select
// FALLBACK: scf.for {{.*}} step %[[ONE1024]] {
// FALLBACK: %[[FBVALUE1024:.*]] = memref.load
// FALLBACK: arith.divf %[[FBVALUE1024]], %[[FBSCALE1024]] : f32
// FALLBACK-NOT: arith.mulf
// FALLBACK: return
