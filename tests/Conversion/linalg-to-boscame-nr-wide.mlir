// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga nr-tile-n=16' --canonicalize | FileCheck %s -DTILE_N=16
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga nr-tile-n=32' --canonicalize | FileCheck %s -DTILE_N=32
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga nr-tile-n=64' --canonicalize | FileCheck %s -DTILE_N=64
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga' > %t.default
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga nr-tile-n=16' > %t.explicit
// RUN: diff %t.default %t.explicit
// RUN: not buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga nr-tile-n=128' 2>&1 | FileCheck %s --check-prefix=SIZE
// RUN: not buddy-opt %s --lower-linalg-to-boscame='target=upstream nr-tile-n=32' 2>&1 | FileCheck %s --check-prefix=TARGET
// RUN: not buddy-opt %s --lower-linalg-to-boscame='target=qwen3-fpga nr-tile-n=64' 2>&1 | FileCheck %s --check-prefix=TARGET

// SIZE: error: nr-tile-n must be 16, 32, or 64
// TARGET: error: non-default nr-tile-n requires target=nr-fpga

// M17/N67/K65 has a tail in every axis for all three N schedules. The scratch
// must grow with N; C keeps its original row stride (67 * sizeof(i32)). Loading
// C once outside the full-K loop preserves nonzero incoming accumulators.
// CHECK-LABEL: func.func @packed_tails
// CHECK-DAG: %[[STRIDE_C:.*]] = arith.constant 268 : i64
// CHECK-DAG: %[[N:.*]] = arith.constant [[TILE_N]] : index
// CHECK: %[[SCRATCH:.*]] = memref.alloca() {{.*}} : memref<[[TILE_N]]x64xi8>
// CHECK: llvm.fence seq_cst
// CHECK: scf.for %[[MI:.*]] = {{.*}} {
// CHECK: scf.for %[[NI:.*]] = {{.*}} step %[[N]] {
// CHECK: %[[REMAIN_N:.*]] = arith.subi {{.*}}, %[[NI]] : index
// CHECK: %[[COLS:.*]] = arith.minsi %[[REMAIN_N]], %[[N]] : index
// CHECK: %[[C:.*]] = memref.subview %arg2[%[[MI]], %[[NI]]] [{{.*}}, %[[COLS]]]
// CHECK: bosc_ame.msettilen
// CHECK: %[[SEED:.*]] = bosc_ame.mlce32.m %[[C]], %[[STRIDE_C]]
// CHECK-NOT: bosc_ame.msce32
// CHECK: %[[RESULT:.*]] = scf.for %[[KI:.*]] = {{.*}} step %c64 iter_args(%[[ACC:.*]] = %[[SEED]])
// CHECK: %[[REMAIN_K:.*]] = arith.subi {{.*}}, %[[KI]] : index
// CHECK: %[[DEPTH:.*]] = arith.minsi %[[REMAIN_K]], %c64 : index
// CHECK: scf.for %[[PN:.*]] = {{.*}} to %[[COLS]] step {{.*}} {
// CHECK: scf.for %[[PK:.*]] = {{.*}} to %[[DEPTH]] step {{.*}} {
// CHECK: memref.store {{.*}}, %[[SCRATCH]][%[[PN]], %[[PK]]] : memref<[[TILE_N]]x64xi8>
// CHECK: bosc_ame.msettilek
// CHECK: bosc_ame.mlbe8.m %[[SCRATCH]]
// CHECK: %[[NEXT:.*]] = bosc_ame.mqma.b.mm %[[ACC]],
// CHECK: scf.yield %[[NEXT]]
// CHECK: bosc_ame.msce32.m %[[RESULT]], %[[C]], %[[STRIDE_C]]
// CHECK: llvm.fence seq_cst
// CHECK: return
func.func @packed_tails(%a: memref<17x65xi8>, %b: memref<65x67xi8>,
                        %c: memref<17x67xi32>) {
  linalg.matmul ins(%a, %b : memref<17x65xi8>, memref<65x67xi8>)
      outs(%c : memref<17x67xi32>)
  return
}

// Physical [N,K] rows need no packing, including padded row strides and
// offsets. Loading B must still use its full row stride, not the tile depth.
// CHECK-LABEL: func.func @physical_nk
// CHECK-DAG: %[[STRIDE_B:.*]] = arith.constant 1040 : i64
// CHECK-DAG: %[[STRIDE_C:.*]] = arith.constant 280 : i64
// CHECK-DAG: %[[N:.*]] = arith.constant [[TILE_N]] : index
// CHECK-NOT: memref.alloca
// CHECK: scf.for %[[NI:.*]] = {{.*}} step %[[N]] {
// CHECK: %[[REMAIN:.*]] = arith.subi {{.*}}, %[[NI]]
// CHECK: %[[COLS:.*]] = arith.minsi %[[REMAIN]], %[[N]]
// CHECK: bosc_ame.mlce32.m {{.*}}, %[[STRIDE_C]]
// CHECK: scf.for %[[KI:.*]] = {{.*}} step %c64 iter_args
// CHECK: %[[B:.*]] = memref.subview %arg1[%[[NI]], %[[KI]]] [%[[COLS]], {{.*}}]
// CHECK: bosc_ame.mlbe8.m %[[B]], %[[STRIDE_B]]
// CHECK: bosc_ame.mqma.b.mm
// CHECK: bosc_ame.msce32.m {{.*}}, %[[STRIDE_C]]
// CHECK: return
func.func @physical_nk(%a: memref<1x1024xi8, strided<[1056, 1], offset: 7>>,
                       %b: memref<67x1024xi8, strided<[1040, 1], offset: 5>>,
                       %c: memref<1x67xi32, strided<[70, 1], offset: 3>>) {
  linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>,
      affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>]
      ins(%a, %b : memref<1x1024xi8, strided<[1056, 1], offset: 7>>,
          memref<67x1024xi8, strided<[1040, 1], offset: 5>>)
      outs(%c : memref<1x67xi32, strided<[70, 1], offset: 3>>)
  return
}

// A [K,N] transposed-stride view of the same physical B rows is also direct.
// CHECK-LABEL: func.func @transposed_view
// CHECK: %[[STRIDE:.*]] = arith.constant 1024 : i64
// CHECK-NOT: memref.alloca
// CHECK: bosc_ame.mlbe8.m {{.*}}, %[[STRIDE]]
// CHECK: return
func.func @transposed_view(%a: memref<16x1024xi8>,
                           %b: memref<1024x67xi8, strided<[1, 1024]>>,
                           %c: memref<16x67xi32>) {
  linalg.matmul ins(%a, %b : memref<16x1024xi8>,
      memref<1024x67xi8, strided<[1, 1024]>>) outs(%c : memref<16x67xi32>)
  return
}
