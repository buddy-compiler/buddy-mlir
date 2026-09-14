// RUN: buddy-opt %s \
// RUN:   '--lower-linalg-to-boscame=target=qwen3-fpga triton-w8a8-fast-path=true' \
// RUN:   --canonicalize | FileCheck %s --check-prefix=FAST
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=qwen3-fpga' | \
// RUN:   FileCheck %s --check-prefix=FALLBACK

// Triton lowers tl.dot(i8, i8) to an exact i32 linalg.matmul followed by an
// elementwise sitofp.  The opt-in AME path fuses that pair and consumes the
// physical [N, full-K] weight through its logical [K, N] strided view.

func.func @decode_n128(
    %a: memref<1x64xi8>,
    %b: memref<64x128xi8, strided<[1, 64]>>) -> f32 {
  %zero_i32 = arith.constant 0 : i32
  %dot = memref.alloc() : memref<1x128xi32>
  linalg.fill ins(%zero_i32 : i32) outs(%dot : memref<1x128xi32>)
  linalg.matmul {cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : memref<1x64xi8>,
                    memref<64x128xi8, strided<[1, 64]>>)
      outs(%dot : memref<1x128xi32>)
  %result = memref.alloc() : memref<1x128xf32>
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%dot : memref<1x128xi32>) outs(%result : memref<1x128xf32>) {
    ^bb0(%in: i32, %out: f32):
      %cast = arith.sitofp %in : i32 to f32
      linalg.yield %cast : f32
  }
  %c0 = arith.constant 0 : index
  %final = memref.alloc() : memref<1x128xf32>
  memref.copy %result, %final : memref<1x128xf32> to memref<1x128xf32>
  %value = memref.load %final[%c0, %c0] : memref<1x128xf32>
  return %value : f32
}

func.func @prefill_m32_n64(
    %a: memref<32x64xi8>,
    %b: memref<64x64xi8, strided<[1, 64]>>) -> f32 {
  %zero_i32 = arith.constant 0 : i32
  %zero_template = memref.alloc() : memref<32x64xi32>
  linalg.fill ins(%zero_i32 : i32)
      outs(%zero_template : memref<32x64xi32>)
  %dot = memref.alloc() : memref<32x64xi32>
  memref.copy %zero_template, %dot
      : memref<32x64xi32> to memref<32x64xi32>
  linalg.matmul {cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : memref<32x64xi8>,
                    memref<64x64xi8, strided<[1, 64]>>)
      outs(%dot : memref<32x64xi32>)
  %result = memref.alloc() : memref<32x64xf32>
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%dot : memref<32x64xi32>) outs(%result : memref<32x64xf32>) {
    ^bb0(%in: i32, %out: f32):
      %cast = arith.sitofp %in : i32 to f32
      linalg.yield %cast : f32
  }
  %c0 = arith.constant 0 : index
  %value = memref.load %result[%c0, %c0] : memref<32x64xf32>
  return %value : f32
}

// K > 1024 is intentionally not fused: an arbitrary signed-i8 dot may no
// longer be represented exactly by the fp32 result written from AME.
func.func @k2048_is_not_fused(
    %a: memref<1x2048xi8>,
    %b: memref<2048x64xi8, strided<[1, 2048]>>) -> f32 {
  %zero_i32 = arith.constant 0 : i32
  %dot = memref.alloc() : memref<1x64xi32>
  linalg.fill ins(%zero_i32 : i32) outs(%dot : memref<1x64xi32>)
  linalg.matmul {cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : memref<1x2048xi8>,
                    memref<2048x64xi8, strided<[1, 2048]>>)
      outs(%dot : memref<1x64xi32>)
  %result = memref.alloc() : memref<1x64xf32>
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%dot : memref<1x64xi32>) outs(%result : memref<1x64xf32>) {
    ^bb0(%in: i32, %out: f32):
      %cast = arith.sitofp %in : i32 to f32
      linalg.yield %cast : f32
  }
  %c0 = arith.constant 0 : index
  %value = memref.load %result[%c0, %c0] : memref<1x64xf32>
  return %value : f32
}

// FAST-LABEL: func.func @decode_n128
// FAST-NOT: linalg.matmul
// FAST-NOT: arith.sitofp
// FAST: bosc_ame.msettilem
// FAST: bosc_ame.msettilen
// FAST: bosc_ame.msettilek
// FAST-COUNT-8: bosc_ame.mlce32.m
// The eight accumulator chains are loop-carried SSA values, so the schedule is
// visible as dataflow: one activation tile feeds both N64 half-blocks, and each
// MMA consumes exactly the B tile that was loaded for it.
// FAST: %[[A:.*]] = bosc_ame.mlae8.m
// FAST: %[[B0:.*]] = bosc_ame.mlbe8.m
// FAST: %[[B1:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A]], %[[B0]]
// FAST: %[[B2:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A]], %[[B1]]
// FAST: %[[B3:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A]], %[[B2]]
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A]], %[[B3]]
// FAST: %[[C0:.*]] = bosc_ame.mlbe8.m
// FAST: %[[C1:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A]], %[[C0]]
// FAST: %[[C2:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A]], %[[C1]]
// FAST: %[[C3:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A]], %[[C2]]
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A]], %[[C3]]
// FAST-COUNT-8: bosc_ame.msce32.m
// FAST: llvm.fence seq_cst

// FAST-LABEL: func.func @prefill_m32_n64
// FAST-NOT: linalg.matmul
// FAST-NOT: arith.sitofp
// FAST: bosc_ame.msettilem
// FAST: bosc_ame.msettilen
// FAST: bosc_ame.msettilek
// FAST-COUNT-8: bosc_ame.mlce32.m
// 2A4B: two activation tiles against one bank of four B tiles.  Both A banks
// consume the same B values, which is what makes four B registers enough.
// FAST: %[[A0:.*]] = bosc_ame.mlae8.m
// FAST: %[[A1:.*]] = bosc_ame.mlae8.m
// FAST: %[[B0:.*]] = bosc_ame.mlbe8.m
// FAST: %[[B1:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A0]], %[[B0]]
// FAST: %[[B2:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A1]], %[[B0]]
// FAST: %[[B3:.*]] = bosc_ame.mlbe8.m
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A0]], %[[B1]]
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A1]], %[[B1]]
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A0]], %[[B2]]
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A1]], %[[B2]]
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A0]], %[[B3]]
// FAST: bosc_ame.mqma.b.mm %{{.*}}, %[[A1]], %[[B3]]
// FAST-COUNT-8: bosc_ame.msce32.m
// FAST: llvm.fence seq_cst

// FAST-LABEL: func.func @k2048_is_not_fused
// FAST: linalg.matmul
// FAST: arith.sitofp

// FALLBACK-LABEL: func.func @decode_n128
// FALLBACK: linalg.matmul
// FALLBACK: arith.sitofp
// FALLBACK-LABEL: func.func @prefill_m32_n64
// FALLBACK: linalg.matmul
// FALLBACK: arith.sitofp
// FALLBACK-LABEL: func.func @k2048_is_not_fused
// FALLBACK: linalg.matmul
// FALLBACK: arith.sitofp
