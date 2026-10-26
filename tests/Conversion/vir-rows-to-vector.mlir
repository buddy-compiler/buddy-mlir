// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=1024' \
// RUN: | FileCheck %s --check-prefix=FIXED
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s --check-prefix=SCALABLE
// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=48' \
// RUN: | FileCheck %s --check-prefix=W48

// In a vir.set_vl with results, a !vir.vec<2x?xf32> has its two rows of vl
// lanes one after the other, in 2 * vl lanes. Its rows are the dims before
// the last one of an accessed memref (as for vector.load), collapsed into the
// last one: a row of k is one access of 128 lanes, the dot product one
// reduction.
func.func @dot(%q: memref<2x64xf32>,
               %k: memref<8x2x64xf32, strided<[256, 64, 1], offset: ?>>,
               %o: memref<2x64xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %q[] : memref<2x64xf32> -> !vir.vec<2x?xf32>
    %y = vir.load %k[%c3, %c0, %c0] : memref<8x2x64xf32, strided<[256, 64, 1], offset: ?>> -> !vir.vec<2x?xf32>
    %m = arith.mulf %x, %y : !vir.vec<2x?xf32>
    vir.store %m, %o[] : !vir.vec<2x?xf32> -> memref<2x64xf32>
    %s = vir.reduce %m, %z {kind = "add"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// Six vectors of 128 lanes at most at VLEN 1024 (LMUL 4), clamped from 256.
// FIXED-LABEL: func.func @dot
// FIXED: %[[Q:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1]] : memref<2x64xf32> into memref<128xf32>
// FIXED: %[[X:.*]] = vector.load %[[Q]][%{{.*}}] : memref<128xf32>, vector<128xf32>
// FIXED: %[[K:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0], [1, 2]] : memref<8x2x64xf32, strided<[256, 64, 1], offset: ?>> into memref<8x128xf32, strided<[256, 1], offset: ?>>
// FIXED: %[[ROW:.*]] = arith.muli %c0{{.*}}, %c64{{.*}} : index
// FIXED: %[[COL:.*]] = arith.addi %[[ROW]], %c0{{.*}} : index
// FIXED: %[[AT:.*]] = arith.addi %[[COL]], %{{.*}} : index
// FIXED: %[[Y:.*]] = vector.load %[[K]][%c3, %[[AT]]] : memref<8x128xf32, strided<[256, 1], offset: ?>>, vector<128xf32>
// FIXED: %[[M:.*]] = arith.mulf %[[X]], %[[Y]] : vector<128xf32>
// FIXED: vector.store %[[M]], %{{.*}} : memref<128xf32>, vector<128xf32>
// FIXED: vector.reduction <add>, %[[M]], %{{.*}} : vector<128xf32> into f32
// FIXED-NOT: vector.reduction

// At VLEN 256: two vector<[16]xf32> of 64 lanes, masked from 128 lanes on.
// SCALABLE-LABEL: func.func @dot
// SCALABLE: %[[LANES:.*]] = arith.constant 128 : index
// SCALABLE: arith.subi %[[LANES]], %{{.*}} : index
// SCALABLE: vector.create_mask %{{.*}} : vector<[16]xi1>
// SCALABLE: arith.subi %[[LANES]], %{{.*}} : index
// SCALABLE: vector.create_mask %{{.*}} : vector<[16]xi1>
// SCALABLE-NOT: vector.create_mask
// SCALABLE: vector.maskedload {{.*}} : memref<128xf32>, vector<[16]xi1>, vector<[16]xf32> into vector<[16]xf32>
// SCALABLE: vector.maskedload {{.*}} : memref<128xf32>, vector<[16]xi1>, vector<[16]xf32> into vector<[16]xf32>
// SCALABLE: vector.maskedload {{.*}} : memref<8x128xf32, strided<[256, 1], offset: ?>>, vector<[16]xi1>, vector<[16]xf32> into vector<[16]xf32>

// 128 lanes at width 48: two vectors and 32 scalars.
// W48-LABEL: func.func @dot
// W48-COUNT-2: vector.load %{{.*}} : memref<128xf32>, vector<48xf32>
// W48-COUNT-32: memref.load %{{.*}} : memref<128xf32>
// W48: vector.load %{{.*}} : memref<8x128xf32, strided<[256, 1], offset: ?>>, vector<48xf32>

// A loop carries the vectors of the rows.
func.func @acc(%k: memref<8x2x64xf32>, %w: memref<8xf32>, %o: memref<2x64xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %a0 = vir.broadcast %z : f32 -> !vir.vec<2x?xf32>
    %acc = scf.for %i = %c0 to %c8 step %c1 iter_args(%a = %a0) -> (!vir.vec<2x?xf32>) {
      %kv = vir.load %k[%i, %c0, %c0] : memref<8x2x64xf32> -> !vir.vec<2x?xf32>
      %ws = memref.load %w[%i] : memref<8xf32>
      %wb = vir.broadcast %ws : f32 -> !vir.vec<2x?xf32>
      %n = vir.fma %kv, %wb, %a : !vir.vec<2x?xf32>
      scf.yield %n : !vir.vec<2x?xf32>
    }
    vir.store %acc, %o[] : !vir.vec<2x?xf32> -> memref<2x64xf32>
    %s = vir.reduce %acc, %z {kind = "maxnum"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// FIXED-LABEL: func.func @acc
// FIXED: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<128xf32>) {
// FIXED: vector.load %{{.*}} : memref<8x128xf32>, vector<128xf32>
// FIXED: vector.fma {{.*}} : vector<128xf32>
// SCALABLE-LABEL: func.func @acc
// SCALABLE: scf.for {{.*}} -> (vector<[16]xf32>, vector<[16]xf32>) {
