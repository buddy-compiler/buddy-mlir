// RUN: buddy-opt %s '--lower-linalg-to-boscame=target=qwen3-fpga triton-w8a8-fast-path=true' \
// RUN:   | FileCheck %s
//
// Multi-tile K reduction on the FPGA target.  The accumulator has to stay
// resident for every K tile, which in the value-semantics layer means the
// MMA result is a loop-carried SSA value: the chain is `mlce32.m` seed ->
// `scf.for` iter_arg -> MMA -> `scf.yield` -> `msce32.m`.  Reloading the fp32
// destination between K tiles would reinterpret its bits as an integer
// accumulator, so the loop structure is part of the correctness contract.
module {

  // 4x128 * 128x4: the 16x64 hardware tile needs two K iterations.
  func.func @k128_single_chain(%A: memref<4x128xi8>,
                               %B: memref<128x4xi8>,
                               %out: memref<4x4xf32>) {
    // The caller guarantees that input and output buffers do not overlap.
    %A_distinct, %B_distinct, %out_distinct = memref.distinct_objects %A, %B, %out : memref<4x128xi8>, memref<128x4xi8>, memref<4x4xf32>

    %C = memref.alloc() : memref<4x4xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%C : memref<4x4xf32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
        ins(%A_distinct, %B_distinct : memref<4x128xi8>, memref<128x4xi8>)
        outs(%C : memref<4x4xf32>)
    memref.copy %C, %out_distinct : memref<4x4xf32> to memref<4x4xf32>
    memref.dealloc %C : memref<4x4xf32>
    return
  }

  // 32x192 * 192x64 with a physically transposed weight: the wide 2A4B
  // schedule, eight accumulator chains across three K iterations.
  func.func @k192_2a4b_eight_chains(%A: memref<32x192xi8>,
                                    %B: memref<192x64xi8, strided<[1, 192]>>,
                                    %out: memref<32x64xf32>) {
    // The caller guarantees that input and output buffers do not overlap.
    %A_distinct, %B_distinct, %out_distinct = memref.distinct_objects %A, %B, %out : memref<32x192xi8>, memref<192x64xi8, strided<[1, 192]>>, memref<32x64xf32>

    %C = memref.alloc() : memref<32x64xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%C : memref<32x64xf32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
        ins(%A_distinct, %B_distinct : memref<32x192xi8>, memref<192x64xi8, strided<[1, 192]>>)
        outs(%C : memref<32x64xf32>)
    memref.copy %C, %out_distinct : memref<32x64xf32> to memref<32x64xf32>
    memref.dealloc %C : memref<32x64xf32>
    return
  }

  // 4x1024 * 1024x4: sixteen K tiles on a single accumulator chain.  This is the
  // largest K the Triton fusion accepts; the direct-C path has no smaller limit,
  // so the boundary also checks that a long reduction keeps exactly one SSA chain
  // and one final write-back.
  func.func @k1024_single_chain(%A: memref<4x1024xi8>,
                                %B: memref<1024x4xi8>,
                                %out: memref<4x4xf32>) {
    // The caller guarantees that input and output buffers do not overlap.
    %A_distinct, %B_distinct, %out_distinct = memref.distinct_objects %A, %B, %out : memref<4x1024xi8>, memref<1024x4xi8>, memref<4x4xf32>

    %C = memref.alloc() : memref<4x4xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%C : memref<4x4xf32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
        ins(%A_distinct, %B_distinct : memref<4x1024xi8>, memref<1024x4xi8>)
        outs(%C : memref<4x4xf32>)
    memref.copy %C, %out_distinct : memref<4x4xf32> to memref<4x4xf32>
    memref.dealloc %C : memref<4x4xf32>
    return
  }
}

// CHECK-LABEL: func.func @k128_single_chain
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// The accumulator seed is loaded once, before the reduction.
// CHECK: bosc_ame.msettype %{{.*}}65602
// CHECK: %[[SEED:.*]] = bosc_ame.mlce32.m {{.*}} -> vector<4x4xi32>
// CHECK: bosc_ame.msettype %{{.*}}65552
// One chain: one iter_arg, one yielded value, and the MMA consumes the value
// carried by the loop rather than a freshly loaded destination.
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[CARRY:.*]] = %[[SEED]]) -> (vector<4x4xi32>) {
// CHECK:   bosc_ame.msettilek
// CHECK:   bosc_ame.mlae8.m
// CHECK:   bosc_ame.mlbte8.m
// CHECK:   %[[NEXT:.*]] = bosc_ame.mqma.b.mm %[[CARRY]], {{.*}} : vector<4x4xi32>, vector<4x4xi8>, vector<4x4xi8> -> vector<4x4xi32>
// CHECK:   scf.yield %[[NEXT]] : vector<4x4xi32>
// CHECK: }
// The write-back uses the reduction result, never a reload of the destination.
// CHECK: bosc_ame.msettype %{{.*}}65602
// CHECK: bosc_ame.msce32.m %[[LOOP]], {{.*}} : vector<4x4xi32>, memref<{{.*}}xf32

// CHECK-LABEL: func.func @k192_2a4b_eight_chains
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// Eight chains: eight seeds, eight iter_args, eight yielded values.
// CHECK: bosc_ame.msettype %{{.*}}65602
// CHECK-COUNT-8: bosc_ame.mlce32.m {{.*}} -> vector<4x4xi32>
// CHECK: bosc_ame.msettype %{{.*}}65552
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = {{.*}}, %{{.*}} = {{.*}}, %{{.*}} = {{.*}}, %{{.*}} = {{.*}}, %{{.*}} = {{.*}}, %{{.*}} = {{.*}}, %{{.*}} = {{.*}}, %{{.*}} = {{.*}}) -> (vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>) {
// CHECK:   scf.yield {{.*}} : vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>, vector<4x4xi32>
// CHECK: }
// CHECK: bosc_ame.msettype %{{.*}}65602
// CHECK-COUNT-8: bosc_ame.msce32.m
// CHECK: return

// K=1024: the same single chain must span all sixteen K tiles; only the end of
// the reduction writes the destination.
// CHECK-LABEL: func.func @k1024_single_chain
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: bosc_ame.msettype %{{.*}}65602
// CHECK: %[[SEED1024:.*]] = bosc_ame.mlce32.m {{.*}} -> vector<4x4xi32>
// CHECK: bosc_ame.msettype %{{.*}}65552
// CHECK: %[[LOOP1024:.*]] = scf.for {{.*}} iter_args(%[[CARRY1024:.*]] = %[[SEED1024]]) -> (vector<4x4xi32>) {
// CHECK:   bosc_ame.msettilek
// CHECK:   bosc_ame.mlae8.m {{.*}} -> vector<4x4xi8>
// CHECK:   bosc_ame.mlbte8.m {{.*}} -> vector<4x4xi8>
// CHECK:   %[[NEXT1024:.*]] = bosc_ame.mqma.b.mm %[[CARRY1024]], {{.*}} : vector<4x4xi32>, vector<4x4xi8>, vector<4x4xi8> -> vector<4x4xi32>
// CHECK:   scf.yield %[[NEXT1024]] : vector<4x4xi32>
// CHECK: }
// CHECK: bosc_ame.msettype %{{.*}}65602
// CHECK: bosc_ame.msce32.m %[[LOOP1024]], {{.*}} : vector<4x4xi32>, memref<{{.*}}xf32
