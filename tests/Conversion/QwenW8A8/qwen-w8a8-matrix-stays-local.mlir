// RUN: buddy-opt %s '--lower-linalg-to-boscame=target=qwen3-fpga triton-w8a8-fast-path=true' \
// RUN:   --lower-bosc-ame | FileCheck %s
//
// ABI invariant of the FPGA pathway: matrix values stay function-local.  Tiles
// and accumulators are produced by loads from memrefs and consumed by stores to
// memrefs, so no function signature may carry a matrix type - the roles only
// exist inside a function body.  The role ABI proposal
// (docs/BOSCAMEFPGARoleABI.md) depends on this property: a role-carrying type at
// a call boundary would need a calling convention that this lowering does not
// define.
//
// CHECK-NOT: func.func @{{.*}}vector<4x4xi
// CHECK: func.func @k128_single_chain
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
}
