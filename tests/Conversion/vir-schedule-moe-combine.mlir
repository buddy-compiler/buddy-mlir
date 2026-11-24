// RUN: buddy-opt %s -lower-linalg-to-vir \
// RUN:     -lower-vir-to-vector='rvv-vlen=256 verbose=true' \
// RUN: | FileCheck %s --check-prefix=NOSCHED
// RUN: buddy-opt %s -lower-linalg-to-vir \
// RUN:     -vir-schedule='rvv=true verbose=true' \
// RUN:     -lower-vir-to-vector='rvv-vlen=256 verbose=true' \
// RUN: | FileCheck %s --check-prefix=SCHED

// The combine of a mixture-of-experts layer at decode: the token's output is
// the sum of the outputs of its top-8 experts weighted by their gates
// (DeepSeek-V3 and R1: 8 routed experts per token, hidden size 7168).
// -lower-linalg-to-vir loads the 8 inputs of the generic before its body:
// all 8 are live at the first product, 9 dynamic vectors with it, and the
// vectors are sized for 9: LMUL 2, 16 f32 lanes at VLEN 256. (A gate is a
// broadcast scalar, the scalar operand of vfmul.vf on RVV, in a scalar
// register: neither -lower-vir-to-vector nor -vir-schedule with rvv counts
// it.) -vir-schedule loads each expert's output where it is used: 3 live at
// most (the sum, the output and its product), LMUL 8, 64 lanes, 4 times the
// lanes per instruction. The ops are the same, in another order: the
// results are bit for bit the same (no reduction here).
func.func @moe_combine_top8(
    %e0: memref<7168xf32>, %e1: memref<7168xf32>, %e2: memref<7168xf32>,
    %e3: memref<7168xf32>, %e4: memref<7168xf32>, %e5: memref<7168xf32>,
    %e6: memref<7168xf32>, %e7: memref<7168xf32>,
    %g0: f32, %g1: f32, %g2: f32, %g3: f32,
    %g4: f32, %g5: f32, %g6: f32, %g7: f32,
    %y: memref<7168xf32>) {
  linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%e0, %e1, %e2, %e3, %e4, %e5, %e6, %e7
          : memref<7168xf32>, memref<7168xf32>, memref<7168xf32>,
            memref<7168xf32>, memref<7168xf32>, memref<7168xf32>,
            memref<7168xf32>, memref<7168xf32>)
      outs(%y : memref<7168xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %x5: f32,
       %x6: f32, %x7: f32, %out: f32):
    %p0 = arith.mulf %x0, %g0 : f32
    %p1 = arith.mulf %x1, %g1 : f32
    %s1 = arith.addf %p0, %p1 : f32
    %p2 = arith.mulf %x2, %g2 : f32
    %s2 = arith.addf %s1, %p2 : f32
    %p3 = arith.mulf %x3, %g3 : f32
    %s3 = arith.addf %s2, %p3 : f32
    %p4 = arith.mulf %x4, %g4 : f32
    %s4 = arith.addf %s3, %p4 : f32
    %p5 = arith.mulf %x5, %g5 : f32
    %s5 = arith.addf %s4, %p5 : f32
    %p6 = arith.mulf %x6, %g6 : f32
    %s6 = arith.addf %s5, %p6 : f32
    %p7 = arith.mulf %x7, %g7 : f32
    %s7 = arith.addf %s6, %p7 : f32
    linalg.yield %s7 : f32
  }
  return
}

// NOSCHED: Final Max Active Vectors (Req_Group): 9
// NOSCHED: Calculated LMUL = 2, Size_vec = (2 * 256) / 32 = 16
// NOSCHED-LABEL: func.func @moe_combine_top8
// NOSCHED-COUNT-8: vector.load {{.*}}, vector<16xf32>
// NOSCHED: arith.mulf {{.*}} : vector<16xf32>

// SCHED: vir-schedule: MaxLive 9 -> 3 in the block of vir.set_vl
// SCHED: Final Max Active Vectors (Req_Group): 3
// SCHED: Calculated LMUL = 8, Size_vec = (8 * 256) / 32 = 64
// SCHED-LABEL: func.func @moe_combine_top8
// SCHED: %[[X0:.*]] = vector.load {{.*}}, vector<64xf32>
// SCHED: %[[P0:.*]] = arith.mulf %[[X0]], {{.*}} : vector<64xf32>
// SCHED: %[[X1:.*]] = vector.load {{.*}}, vector<64xf32>
// SCHED: %[[P1:.*]] = arith.mulf %[[X1]], {{.*}} : vector<64xf32>
// SCHED: arith.addf %[[P0]], %[[P1]] : vector<64xf32>
