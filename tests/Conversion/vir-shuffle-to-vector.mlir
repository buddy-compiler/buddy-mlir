// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=1024' \
// RUN: | FileCheck %s --check-prefix=FIXED
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=1024' \
// RUN: | FileCheck %s --check-prefix=SCALABLE

// RoPE on a head of 128 as !vir.vec<2x?xf32> of vl 64: rotate_half is
// [-x[1], x[0]], a vir.shuffle of the rows of -x and x. Within the one
// vector of the 128 lanes: a constant vector.shuffle of fixed vectors; of
// vector<[8]xf32>, each row a vector<[4]xf32> (64 lanes at VLEN 1024) taken
// and put in place.
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
// SCALABLE-NOT: maskedload
// SCALABLE: %[[X:.*]] = vector.load {{.*}} : memref<128xf32>, vector<[8]xf32>
// SCALABLE: %[[NX:.*]] = arith.negf %[[X]] : vector<[8]xf32>
// SCALABLE-NOT: create_mask
// SCALABLE: %[[NONE:.*]] = ub.poison : vector<[8]xf32>
// SCALABLE: %[[ROW1:.*]] = vector.scalable.extract %[[NX]][4] : vector<[4]xf32> from vector<[8]xf32>
// SCALABLE: %[[HALF:.*]] = vector.scalable.insert %[[ROW1]], %[[NONE]][0] : vector<[4]xf32> into vector<[8]xf32>
// SCALABLE: %[[ROW0:.*]] = vector.scalable.extract %[[X]][0] : vector<[4]xf32> from vector<[8]xf32>
// SCALABLE: %[[ROT:.*]] = vector.scalable.insert %[[ROW0]], %[[HALF]][4] : vector<[4]xf32> into vector<[8]xf32>
// SCALABLE: arith.mulf %[[ROT]], %{{.*}} : vector<[8]xf32>

// Rows of both vectors: vector<[1]xf32> parts of vector<[4]xf32>.
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
// SCALABLE: %[[X:.*]] = vector.load {{.*}} : memref<64xf32>, vector<[4]xf32>
// SCALABLE: %[[Y:.*]] = vector.load {{.*}} : memref<64xf32>, vector<[4]xf32>
// SCALABLE: %[[A:.*]] = vector.scalable.extract %[[X]][1] : vector<[1]xf32> from vector<[4]xf32>
// SCALABLE: %[[W0:.*]] = vector.scalable.insert %[[A]], %{{.*}}[0]
// SCALABLE: %[[B:.*]] = vector.scalable.extract %[[Y]][2]
// SCALABLE: %[[W1:.*]] = vector.scalable.insert %[[B]], %[[W0]][1]
// SCALABLE: %[[C:.*]] = vector.scalable.extract %[[X]][0]
// SCALABLE: %[[W2:.*]] = vector.scalable.insert %[[C]], %[[W1]][2]
// SCALABLE: %[[D:.*]] = vector.scalable.extract %[[X]][3]
// SCALABLE: vector.scalable.insert %[[D]], %[[W2]][3]

// A row of 4 lanes is less than vector<[1]xf32> (16 lanes at VLEN 1024): in
// the one vector<[1]xf32> of the 16 lanes, rows slid up and down and
// selected.
func.func @perm_small(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %o: memref<4x4xf32>) -> f32 {
  %vl = arith.constant 4 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<4x4xf32> -> !vir.vec<4x?xf32>
    %y = vir.load %b[] : memref<4x4xf32> -> !vir.vec<4x?xf32>
    %w = vir.shuffle %x, %y [1, 6, 0, 3] : !vir.vec<4x?xf32>
    vir.store %w, %o[] : !vir.vec<4x?xf32> -> memref<4x4xf32>
    %d = vir.reduce %w, %z {kind = "add"} : !vir.vec<4x?xf32>, f32 -> f32
    vector.yield %d : f32
  } -> f32
  return %r : f32
}

// SCALABLE-LABEL: func.func @perm_small
// SCALABLE: %[[X:.*]] = vector.load {{.*}} : memref<16xf32>, vector<[1]xf32>
// SCALABLE: %[[Y:.*]] = vector.load {{.*}} : memref<16xf32>, vector<[1]xf32>
// SCALABLE: %[[R0:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[X]], %{{.*}}, %{{.*}})
// SCALABLE: %[[R1:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[Y]], %{{.*}}, %{{.*}})
// SCALABLE: %[[M1:.*]] = vector.create_mask %c4{{.*}}
// SCALABLE: %[[S1:.*]] = arith.select %[[M1]], %[[R0]], %[[R1]]
// SCALABLE: %[[R2:.*]] = llvm.call_intrinsic "llvm.vector.splice.right"(%{{.*}}, %[[X]], %{{.*}})
// SCALABLE: %[[M2:.*]] = vector.create_mask %c8{{.*}}
// SCALABLE: %[[S2:.*]] = arith.select %[[M2]], %[[S1]], %[[R2]]
// SCALABLE: %[[M3:.*]] = vector.create_mask %c12{{.*}}
// SCALABLE: arith.select %[[M3]], %[[S2]], %[[X]]

// The splice intrinsic takes LLVM types: index elements are cast to
// integers of the index width around it.
func.func @perm_small_index(%a: memref<4x4xindex>, %o: memref<4x4xindex>) -> i32 {
  %vl = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<4x4xindex> -> !vir.vec<4x?xindex>
    %y = vir.shuffle %x, %x [1, 0, 3, 2] : !vir.vec<4x?xindex>
    vir.store %y, %o[] : !vir.vec<4x?xindex> -> memref<4x4xindex>
    vector.yield %zero : i32
  } -> i32
  return %r : i32
}

// SCALABLE-LABEL: func.func @perm_small_index
// SCALABLE: %[[X:.*]] = vector.load {{.*}} : memref<16xindex>, vector<[1]xindex>
// SCALABLE: %[[I:.*]] = arith.index_cast %[[X]] : vector<[1]xindex> to vector<[1]xi64>
// SCALABLE: %[[S:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[I]], %{{.*}}, %{{.*}}) : (vector<[1]xi64>, vector<[1]xi64>, i32) -> vector<[1]xi64>
// SCALABLE: arith.index_cast %[[S]] : vector<[1]xi64> to vector<[1]xindex>
