// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga' --canonicalize | FileCheck %s --check-prefix=NR --implicit-check-not=mlbte --implicit-check-not=msettypei
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga' --lower-bosc-ame | FileCheck %s --check-prefix=EXPORT
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga' --lower-bosc-ame --lower-bosc-ame | FileCheck %s --check-prefix=EXPORT
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga' --lower-bosc-ame \
// RUN:   --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
// RUN:   --expand-strided-metadata --lower-affine --convert-cf-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-math-to-llvm --convert-func-to-llvm \
// RUN:   --finalize-memref-to-llvm --reconcile-unrealized-casts | \
// RUN:   buddy-translate --buddy-to-llvmir > %t.ll
// RUN: llc %t.ll -O2 -verify-machineinstrs -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,-v,+xboscame -o - | FileCheck %s --check-prefix=ASM --implicit-check-not=mlbte

// A tail in every axis exercises the clamped tile sizes and scratch packing.
// C is an incoming integer accumulator; it must not be zeroed or converted.
func.func @row_major_tails(%a: memref<17x65xi8>, %b: memref<65x19xi8>,
                           %c: memref<17x19xi32>) {
  linalg.matmul ins(%a, %b : memref<17x65xi8>, memref<65x19xi8>)
      outs(%c : memref<17x19xi32>)
  return
}

// NR-LABEL: func.func @row_major_tails
// NR: memref.alloca() {{.*}} : memref<16x64xi8>
// NR: llvm.fence seq_cst
// NR: bosc_ame.mlce32.m {{.*}} : memref<?x?xi32
// NR: scf.for {{.*}} iter_args({{.*}}) -> (vector<4x4xi32>)
// NR: scf.for
// NR: scf.for
// NR: memref.load
// NR: memref.store
// NR: llvm.fence seq_cst
// NR: bosc_ame.mlbe8.m {{.*}} : memref<16x64xi8>
// NR: bosc_ame.mqma.b.mm
// NR: scf.yield {{.*}} : vector<4x4xi32>
// NR: bosc_ame.msce32.m {{.*}} : vector<4x4xi32>, memref<?x?xi32
// NR: llvm.fence seq_cst
// NR: return
// EXPORT-LABEL: func.func @row_major_tails
// EXPORT-SAME: +xboscame-fpga
// EXPORT: bosc_ame.intr.fpga.mlce32.m
// EXPORT: bosc_ame.intr.fpga.mlbe8.m
// EXPORT: bosc_ame.intr.fpga.mqma.b.mm
// EXPORT: bosc_ame.intr.fpga.msce32.m
// ASM-LABEL: row_major_tails:
// ASM: mlce32.m acc0
// ASM: mlae8.m tr0
// ASM: mlbe8.m tr4
// ASM: mqma.b.mm acc0, tr0, tr4
// ASM: msce32.m acc0

// A logical [K, N] view of physically [N, K] weights does not need packing.
func.func @decode_transposed_view(%a: memref<1x3072xi8>,
    %b: memref<3072x1024xi8, strided<[1, 3072]>>,
    %c: memref<1x1024xi32>) {
  linalg.matmul ins(%a, %b : memref<1x3072xi8>,
      memref<3072x1024xi8, strided<[1, 3072]>>) outs(%c : memref<1x1024xi32>)
  return
}
// NR-LABEL: func.func @decode_transposed_view
// NR-NOT: memref.alloca
// NR: bosc_ame.mlce32.m
// NR: bosc_ame.mlbe8.m
// NR: bosc_ame.msce32.m
// NR: return

func.func @transpose_b(%a: memref<16x1024xi8>, %b: memref<2048x1024xi8>,
                       %c: memref<16x2048xi32>) {
  linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>,
      affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>]
      ins(%a, %b : memref<16x1024xi8>,
      memref<2048x1024xi8>) outs(%c : memref<16x2048xi32>)
  return
}
// NR-LABEL: func.func @transpose_b
// NR-NOT: memref.alloca
// NR: bosc_ame.mlce32.m
// NR: bosc_ame.mlbe8.m
// NR: bosc_ame.msce32.m
// NR: return

// Attention f32 matmul remains available for the CPU lowering. Selecting NR
// must not reinterpret it as a legacy f32 accumulator store.
func.func @float_cpu(%a: memref<16x128xf32>, %b: memref<128x16xf32>,
                     %c: memref<16x16xf32>) {
  linalg.matmul ins(%a, %b : memref<16x128xf32>, memref<128x16xf32>)
      outs(%c : memref<16x16xf32>)
  return
}
// NR-LABEL: func.func @float_cpu
// NR: linalg.matmul
// NR: return
