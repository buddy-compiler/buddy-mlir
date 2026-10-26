// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=1024' \
// RUN: | FileCheck %s --check-prefix=FIXED
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=1024' \
// RUN: | FileCheck %s --check-prefix=SCALABLE

// RoPE on a head of 128 as !vir.vec<2x?xf32> of vl 64: rotate_half is
// [-x[1], x[0]], a vir.shuffle of the rows of -x and x. Within the one
// vector of the 128 lanes: a constant vector.shuffle of fixed vectors, two
// slides (splice) and a select of scalable ones.
func.func @rope(%q: memref<2x64xf32>, %cs: memref<2x64xf32>,
                %sn: memref<2x64xf32>, %o: memref<2x64xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %q[] : memref<2x64xf32> -> !vir.vec<2x?xf32>
    %c = vir.load %cs[] : memref<2x64xf32> -> !vir.vec<2x?xf32>
    %s = vir.load %sn[] : memref<2x64xf32> -> !vir.vec<2x?xf32>
    %nx = arith.negf %x : !vir.vec<2x?xf32>
    %rot = vir.shuffle %nx, %x [1, 2] : !vir.vec<2x?xf32>
    %t = arith.mulf %rot, %s : !vir.vec<2x?xf32>
    %y = vir.fma %x, %c, %t : !vir.vec<2x?xf32>
    vir.store %y, %o[] : !vir.vec<2x?xf32> -> memref<2x64xf32>
    %d = vir.reduce %y, %z {kind = "add"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %d : f32
  } -> f32
  return %r : f32
}

// FIXED-LABEL: func.func @rope
// FIXED: %[[X:.*]] = vector.load {{.*}} : memref<128xf32>, vector<128xf32>
// FIXED: %[[NX:.*]] = arith.negf %[[X]] : vector<128xf32>
// FIXED: %[[ROT:.*]] = vector.shuffle %[[NX]], %[[X]] [64, 65, {{.*}}, 126, 127, 128, 129, {{.*}}, 190, 191] : vector<128xf32>, vector<128xf32>
// FIXED: arith.mulf %[[ROT]], %{{.*}} : vector<128xf32>

// SCALABLE-LABEL: func.func @rope
// SCALABLE: %[[X:.*]] = vector.maskedload {{.*}} into vector<[8]xf32>
// SCALABLE: %[[NX:.*]] = arith.negf %[[X]] : vector<[8]xf32>
// SCALABLE: %[[BY0:.*]] = arith.constant 64 : i32
// SCALABLE: %[[NONE0:.*]] = ub.poison : vector<[8]xf32>
// SCALABLE: %[[ROW0:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[NX]], %[[NONE0]], %[[BY0]])
// SCALABLE: %[[BY1:.*]] = arith.constant 64 : i32
// SCALABLE: %[[NONE1:.*]] = ub.poison : vector<[8]xf32>
// SCALABLE: %[[ROW1:.*]] = llvm.call_intrinsic "llvm.vector.splice.right"(%[[NONE1]], %[[X]], %[[BY1]])
// SCALABLE: %[[FIRST:.*]] = vector.create_mask %c64{{.*}} : vector<[8]xi1>
// SCALABLE: %[[ROT:.*]] = arith.select %[[FIRST]], %[[ROW0]], %[[ROW1]] : vector<[8]xi1>, vector<[8]xf32>
// SCALABLE: arith.mulf %[[ROT]], %{{.*}} : vector<[8]xf32>

// Rows of both vectors, slid up and down.
func.func @perm(%a: memref<4x16xf32>, %b: memref<4x16xf32>, %o: memref<4x16xf32>) -> f32 {
  %vl = arith.constant 16 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<4x16xf32> -> !vir.vec<4x?xf32>
    %y = vir.load %b[] : memref<4x16xf32> -> !vir.vec<4x?xf32>
    %w = vir.shuffle %x, %y [1, 6, 0, 3] : !vir.vec<4x?xf32>
    vir.store %w, %o[] : !vir.vec<4x?xf32> -> memref<4x16xf32>
    %d = vir.reduce %w, %z {kind = "add"} : !vir.vec<4x?xf32>, f32 -> f32
    vector.yield %d : f32
  } -> f32
  return %r : f32
}

// FIXED-LABEL: func.func @perm
// FIXED: vector.shuffle %{{.*}}, %{{.*}} [16, {{.*}}, 31, 96, {{.*}}, 111, 0, {{.*}}, 15, 48, {{.*}}, 63] : vector<64xf32>, vector<64xf32>

// SCALABLE-LABEL: func.func @perm
// SCALABLE: %[[X:.*]] = vector.maskedload
// SCALABLE: %[[Y:.*]] = vector.maskedload
// SCALABLE: %[[R0:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[X]], %{{.*}}, %{{.*}})
// SCALABLE: %[[R1:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[Y]], %{{.*}}, %{{.*}})
// SCALABLE: %[[M1:.*]] = vector.create_mask %c16{{.*}}
// SCALABLE: %[[S1:.*]] = arith.select %[[M1]], %[[R0]], %[[R1]]
// SCALABLE: %[[R2:.*]] = llvm.call_intrinsic "llvm.vector.splice.right"(%{{.*}}, %[[X]], %{{.*}})
// SCALABLE: %[[M2:.*]] = vector.create_mask %c32{{.*}}
// SCALABLE: %[[S2:.*]] = arith.select %[[M2]], %[[S1]], %[[R2]]
// SCALABLE: %[[M3:.*]] = vector.create_mask %c48{{.*}}
// SCALABLE: arith.select %[[M3]], %[[S2]], %[[X]]
